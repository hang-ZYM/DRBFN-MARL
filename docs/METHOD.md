# 方法详解：DRBFN 算法细节

DRBFN（Dynamic Reward Bayesian Flow Network）按版本的详细算法描述。

> 高层概览见 [README.md](../README.md)。本文档面向希望深入理解实现的读者。

---

## 1. 问题设定

合作型 MARL，共享团队奖励：
- `N` 个智能体在共享环境中
- 每步观测：`(share_obs, obs_1, ..., obs_N)`
- 每步动作：`(a_1, ..., a_N)`
- 团队奖励：`R`（标量；所有智能体相同）
- 目标：训练分解策略 `π(a_i | obs_i)`，最大化 `E[Σ γ^t R_t]`

**信用分配问题**：标准 MAPPO 给每个智能体分配 `R/N`。这在期望上无偏但**方差大**——`t-5` 时刻治疗的 Medivac 与 `t` 时刻完成击杀的 Marine 拿到同样的奖励。DRBFN 学习一个 per-agent 分解 `R → (r_1, ..., r_N)`，更好地反映真实贡献，然后在分解后的奖励上训练 MAPPO。

---

## 2. 架构

### 2.1 BFN 奖励分解器

连续型 **Bayesian Flow Network**（Gao et al., 2023），包括：
- **编码器 φ**：在 `(share_obs, joint_action_onehot)` 上的共享 MLP → 条件嵌入 `h`
- **条件网络**：输入 `(x_t, t, h)` → 预测去噪分布的 `(mean, logvar)`
- **贝叶斯流**：前向噪声过程 `q(x_t | x_0)`，其中 `var_t = σ_min^t`

推理：BFN 从先验 `N(0, 1)` 开始，跑 T 步去噪，以 `(share_obs, joint_action)` 为条件。输出：per-agent 奖励预测 `r_i` 和后验方差 `Var[r_i]`（用作置信度）。

### 2.2 Per-agent Q 网络（Q_i）

MLP，输入 `(share_obs, obs_i, agent_id_onehot)` → 智能体 `i` 所有动作的 Q 值。一个 Q_i 在所有智能体间共享（CTDE；用 agent ID 区分）。

### 2.3 全局 Q 网络（Q_tot）

独立 MLP，输入 `(share_obs, joint_action_onehot)` → 标量 Q_tot。用于锚定团队奖励的一致性。

---

## 3. 训练循环（v1、v3）

DRBFN 的 `train()` 覆盖了 MAPPO：

```
1. 若在 warmup 阶段（total_steps < warmup_t）：
   - 用均匀分解 r_i = R/N 训练 Q_i 和 Q_tot
   - 训练 actor（标准 PPO，奖励为 R/N）
   - 返回（跳过 BFN 训练）

2. 计算门控奖励：
   gated_r_i = confidence · r_i^BFN + (1 - confidence) · (R/N)
   其中 confidence = 1 / (1 + Var[r_i])

3. 仅本次 update 临时替换 buffer.rewards 为 gated_r_i

4. 在门控奖励上跑标准 PPO train()
   → 更新 actor 和 critic

5. 恢复 buffer.rewards 为原始 R

6. 训练 DRBFN 组件：
   - L_Qi：在 gated_r_i 上用 TD 目标训练 Q_i
   - L_Qtot：在 R 上用 TD 目标训练 Q_tot
   - L_BFN：用 Bellman 残差信号训练 BFN
     （v1：1-step；v3：n-step 反事实）
```

### 3.1 Dual Gate（安全保障）

Dual Gate 保证 DRBFN 不会比 MAPPO 表现更差。当 BFN 不确定时（`Var[r_i] → ∞`），`confidence → 0`，`gated_r_i → R/N`（均匀，即原始 MAPPO）。当 BFN 自信时（`Var[r_i] → 0`），`gated_r_i → r_i^BFN`。

这类似于策略蒸馏中的"回退到专家"，但这里的"专家"就是均匀分解。

### 3.2 Warmup 阶段

在前 `warmup_t` 步，BFN 未训练，会产生垃圾输出。Warmup 期间：
- BFN 冻结（不训练）
- Q_i、Q_tot 用 `r_i = R/N`（均匀）训练
- Actor 正常训练，奖励为 `R/N`

Warmup 结束后，BFN 训练激活，此时 Q_i 和 Q_tot 已经有了合理的目标。

---

## 4. 版本差异

### v1：Stick-breaking 分解

**守恒**：结构性硬保证 `Σ r_i = R`。

实现：BFN 输出 `N-1` 个 stick-breaking 权重 `w_i ∈ [0, 1]`，然后：
```
r_1 = w_1 · R
r_2 = (1 - w_1) · w_2 · R
r_3 = (1 - w_1)(1 - w_2) · w_3 · R
...
r_N = Π_{j<N} (1 - w_j) · R  # 最后一个智能体拿到剩余
```

求和精确等于 `R`。

**训练信号（1-step TD）**：
- Q_i 的目标：`r_i + γ · V_i(s')`，其中 `V_i` = max_a Q_i(s', a)
- BFN 的目标：最小化反事实 rollout 中 `(Q_i - r_i_target)` 的方差

**优点**：可证明的守恒；稳定
**缺点**：刚性——无法表达协同效应（即 `Σr_i ≠ R` 更准确的情况）

### v2：生成式范式

**守恒**：通过 loss 项的软先验。

实现：BFN 直接输出 `r_i ∈ ℝ^N`。守恒变为：
```
L_conservation = (Σ r_i - R)^2
total_loss = L_BFN + λ · L_conservation
```

**多样本采样**：`sample_k(K)` 返回 `K` 个 `(r_1, ..., r_N)` 的随机采样。用途：
- 对 `r_i` 估计方差（比 v1 的 stick-breaking 方差更好的置信度信号）
- Q 网络训练的数据增强

**优点**：灵活；软守恒允许协同效应
**缺点**：λ 更难调；不如 v1 稳定

### v3：v1 + n-step 反事实

**基础**：v1（stick-breaking 硬守恒）。

**时间目标**：1-step TD 替换为 **n-step 回报**：
```
G_t^n = Σ_{k=0}^{n-1} γ^k · r_{t+k} + γ^n · V(s_{t+n})
```

**反事实贡献**（智能体 `i`）：
```
Δ_i = G_factual^n - G_counterfactual_i^n
```
其中 `G_counterfactual_i^n` 通过将智能体 `i` 的动作替换为 argmax（来自 Q_i），其他智能体的动作重采样得到。

**为什么用 n-step**：
1. 减少 **bootstrap bias** 传播（1-step TD 有从不完美价值估计累积的偏差）
2. 捕获**延迟贡献**：在 StarCraft 中，Medivac 在 `t-5` 治疗只在 `t` 时刻反映到奖励中。1-step TD 给 Medivac 零信用，n-step 给出正确信用。

**超参数**（默认值见 `config.py`）：
- `drbfn_n_step = 5`
- `drbfn_cf_temperature = 1.0`
- `drbfn_beta = 0.04`（PPO actor 的 KL 系数）
- `drbfn_warmup_t = 10000`

---

## 5. 实现文件

| 版本 | 算法入口 | BFN 模块 |
|---|---|---|
| v1 | `onpolicy/algorithms/r_drbfn/r_drbfn.py` | `onpolicy/algorithms/utils/drbfn.py` |
| v2 | `onpolicy/algorithms/r_drbfn_v2/r_drbfn_v2.py` | `onpolicy/algorithms/utils/drbfn_v2.py` |
| v3 | `onpolicy/algorithms/r_drbfn_v3/r_drbfn_v3.py` | （使用 v1 的 BFN；新训练循环） |

各版本的 Q 网络架构：
| 版本 | Q_i | Q_tot |
|---|---|---|
| v1 | `onpolicy/algorithms/r_drbfn/algorithm/drbfn_nets.py` | 同 |
| v2 | `onpolicy/algorithms/r_drbfn_v2/algorithm/drbfn_nets_v2.py` | 同 |
| v3 | （复用 v1） | （复用 v1） |

---

## 6. 关键超参数

定义在 `onpolicy/config.py` 的 `--drbfn_*` 参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--drbfn_warmup_t` | 10000 | BFN 训练激活前的步数 |
| `--drbfn_n_step` | 5（v3） | n-step 回报长度 |
| `--drbfn_cf_temperature` | 1.0（v3） | 反事实动作采样的 softmax 温度 |
| `--drbfn_beta` | 0.04 | actor 的 KL 系数（R1-Zero 风格用 β=0） |
| `--drbfn_lambda_conservation` | 0.5（v2） | `L_conservation` 的权重 |

---

## 7. 设计动机（为什么用 BFN？）

为什么用 **Bayesian Flow Networks** 做奖励分解，而不是用 MLP 回归器或基于 Q 差的方法（如 QPLEX/MAVEN）？

1. **设计上就是随机的**：BFN 的后验方差免费提供一个有原则的**置信度信号**，我们正好用它做 Dual Gate（安全回退）。
2. **不需要真实奖励标签**：与有监督奖励分解（如基于反事实的手工 `r_i = R - R_{-i}`）不同，BFN 通过下游 Q 学习损失端到端训练。
3. **扎根于生成模型文献**：BFN 在 diffusion/flow 文献中有扎实基础，训练基础设施成熟。

考虑过的备选方案（不在本仓库中）：
- **MAVEN**：潜空间分解；离散 code
- **QPLEX**：基于 simplex 的 Q 分解
- **ROMA**：角色感知分解

DRBFN 的差异在于**分解本身就是模型**，而非辅助输出。BFN 直接输出用于策略训练的 per-agent 奖励。

---

## 参考文献

- Bayesian Flow Networks：Gao et al., 2023. "Bayesian Flow Networks." [arXiv:2308.07037](https://arxiv.org/abs/2308.07037)
- MAPPO：Yu et al., 2022. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." NeurIPS D&B Track.
- QMIX：Rashid et al., 2018. "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning."
- QPLEX：Wang et al., 2021. "QPLEX: Duplex Dueling Multi-Agent Q-Learning."
