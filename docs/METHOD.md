# Method：DRBFN-QVPO 算法细节

> 本文是算法层面的详细描述。代码层面的索引见 [`onpolicy/algorithms/r_drbfn_qvpo/README.md`](../onpolicy/algorithms/r_drbfn_qvpo/README.md)。

---

## 1. 问题设定

合作型 MARL：
- `N` 个智能体共享环境
- 每步观测 `(share_obs, obs_1, ..., obs_N)`
- 每步联合动作 `(a_1, ..., a_N)`
- 团队奖励 `R`（标量，所有智能体相同）
- 目标：最大化 `E[Σ γ^t R_t]`

**信用分配问题**：标准 MAPPO 给每个智能体分配 `R/N`。这在期望上无偏但**高方差**——一个在 `t-5` 时刻治疗的 Medivac 和 `t` 时刻完成击杀的 Marine 拿到同样的奖励，无法区分贡献。

---

## 2. DRBFN-QVPO 的核心思路

### 2.1 用 PBRS 学一个 potential，不直接学 r

经典理论（Ng et al. 1999）证明：**任何形如下式的奖励塑形都不改变最优策略**：

```
r_i(t) = R(t)/N + γ·Φ_i(s_{t+1}, a_{t+1}) - Φ_i(s_t, a_t)
```

其中 Φ_i 是任意函数（potential function）。

这个结果非常强——它给了我们**数学上的安全保证**：无论 Φ 学成什么样，最优策略都不变。最坏情况（Φ=0）退化为标准 MAPPO。

**DRBFN-QVPO 的核心创新**：让 BFN 学一个 Φ(s, a) 的**分布**（不是点估计），通过 Q-加权 VLB 训练，让 Φ 朝"对 Q_tot 反事实敏感"的方向偏移。

### 2.2 为什么用 BFN 学 Φ，而不是 MLP / Diffusion？

1. **后验方差可用**：BFN 的输出是分布，方差提供不确定性信号。部署时用 K-sample argmax 过滤坏样本（见 §4.3）。
2. **变分训练**：BFN 通过变分下界训练，自然支持"加权 VLB"——让某些样本的 log-likelihood 被加权，引导分布朝高对齐度方向偏移。
3. **理论锚定**：BFN 在生成模型文献中有扎实基础（Graves 2023），训练稳定性好。

### 2.3 Q-加权 VLB 的灵感来源

借鉴 QVPO（Ding et al. NeurIPS 2024）的核心 insight：**用 Q 函数的反事实敏感度作为扩散模型/流模型的训练信号**。

QVPO 原本用 diffusion model 做单 agent RL。我们把它迁移到：
- 多智能体场景（每个 agent 一个 potential Φ_i）
- 用 BFN 替代 diffusion（更稳定的生成模型）
- 加上 PBRS 包装（保证策略不变性）

---

## 3. 网络架构

| 网络 | 输入 | 输出 | 训练方式 |
|------|------|------|----------|
| Actor π_i(a_i \| o_i) | o_i | 离散动作分布 | PPO with r_i |
| Q_tot(s, a) | (s, a_joint) | 标量团队 Q | n-step return on R |
| BFN p_φ(Φ \| s, a) | (s, a_joint) | Φ ∈ ℝ^N 分布 | Q-加权 VLB |
| Q_tot target | (s, a_joint) | 标量 | soft update (τ=0.005) |

**注意**：
- 没有 V-critic（Q_tot 兼任）
- 没有 Q_i 网络（per-agent 信号来自 g_i = ∂Q_tot/∂a_i）

---

## 4. 训练循环

### 4.1 总体流程

```
for iteration = 1, 2, ...:
    
    # Phase 1: Rollout（用当前 actor + BFN）
    for t = 1 to T:
        sample a_i ~ π_i(·|o_i)
        execute joint a, observe R, s'
        
        # K-sample argmax 选 r
        if iteration > warmup:
            Φ_star = K_argmax(BFN, s, a, Q_tot, K=K_deploy)
        else:
            Φ_star = 0  # warmup 用 R/N
        
        # 算 r via PBRS（SARSA 风格 a'）
        if t < T:
            r_i = R/N + γ Φ_star_i(s', a') - Φ_star_i(s, a)
        else:
            r_i = R/N - Φ_star_i(s, a)  # terminal
        
        store (s, a, R, s', Φ_star, r) in buffer
    
    # Phase 2: Lower level (PPO)
    A_i = r_i + γ Q_tot(s', a') - Q_tot(s, a)
    PPO update actor
    
    # Phase 3: Q_tot update (on R, not r)
    G_n = Σ_{k=0}^{n-1} γ^k R_{t+k} + γ^n Q_tot_target(s_{t+n}, a_{t+n})
    minimize MSE(Q_tot(s, a), G_n)
    soft update Q_tot_target
    
    # Phase 4: BFN update (Q-weighted VLB)
    if iteration > warmup:
        g_i = Q_tot(s, a_i, a_{-i}) - Q_tot(s, default_i, a_{-i})
        for k = 1..K_train:
            Φ^(k) = sample_phi(h)
            align^(k) = (r^(k) · g_i).sum()
            log_p^(k) = log_prob_phi(Φ^(k), h)
        
        aligns = K_normalize(aligns)
        weights = ReLU(aligns) / sum_per_sa(weights)
        
        L_BFN = -Σ_k weights^(k) · log_p^(k)
        update BFN
```

### 4.2 K-sample argmax 部署

部署时，BFN 采样 K 组 Φ，计算每组的 align = r·g，选 align 最大的 r：

```python
for k in 1..K_deploy:
    phi_k = sample_phi(h).clamp(-phi_clamp, phi_clamp)
    phi_next_k = sample_phi(h_next).clamp(...)
    r_k = R/N + γ·phi_next_k·mask - phi_k
    align_k = (r_k * g_i).sum()
r = r^(argmax_k align_k)
```

**为什么这样**：BFN 是随机生成模型，单个 sample 可能是坏的。K-sample argmax 是一个简单有效的过滤——只接受"对齐度最高"的样本。

### 4.3 g_i 的计算：default action 反事实

```python
q_actual = Q_tot(s, a)
for i in 1..N:
    a_default_i = joint_a.clone()
    a_default_i[:, i, :] = 0
    a_default_i[:, i, default_action] = 1  # SMAC: no-op = 0
    q_default_i = Q_tot(s, a_default_i)
    g_i = q_actual - q_default_i
```

g_i 表示"如果 agent i 改为 default action（在 SMAC 里是 no-op），Q_tot 会下降多少"。这正是 agent i 当前动作对团队价值的贡献。

### 4.4 Q-加权 VLB loss

```python
g_i = compute_g_i(s, a).detach()         # 1 次 forward
h = encode(s, a)                          # 可微

for k in 1..K_train:
    phi = sample_phi(h).clamp(-phi_clamp, phi_clamp)
    phi_next = sample_phi(h_next).clamp(...) (detached)
    r = R/N + γ·phi_next·mask - phi
    align = (r * g_i).sum(dim=-1)
    log_p = log_prob_phi(phi, h)
    aligns.append(align); log_ps.append(log_p)

# K-sample normalize
aligns = (aligns - aligns.mean(dim=0)) / (aligns.std(dim=0) + 1e-8)
weights = F.relu(aligns)  # 只保留正对齐度
if weights.sum() < 1e-6: return  # 全负，跳过

# ⭐ 关键：按 (s,a) 归一化（不是全局！）
weight_per_sa = weights.sum(dim=0, keepdim=True) + 1e-8
weights = weights / weight_per_sa

loss = -(weights * log_ps).sum(dim=0).mean()
```

---

## 5. 关键设计决策（8 项）

| # | 决策 | 实现 | 原因 |
|---|------|------|------|
| 1 | PBRS 形式 a' | buffer 的 a_{t+1}（SARSA） | Wiewiora 等价在 SARSA 下成立 |
| 2 | V-critic | 不要，Q_tot 兼任 | 减少网络数，干净信号 |
| 3 | 离散动作下 g_i | default action 反事实 | 不需要 per-agent Q |
| 4 | BFN σ 防塌缩 | 固定 noise_dev | 跟原 BFN 一致 |
| 5 | Wiewiora 双视角 | 成立（SARSA）→ reward 视角 | 数学等价 |
| 6 | 闭环稳定 | warmup + detach + 监控 | 防止反馈失控 |
| 7 | rollout 选 r | K-sample argmax (K=4) | 过滤坏样本 |
| 8 | align 尺度 | K 内 normalize + Φ clamp=0.3 | 防爆炸 + 公平比较 |

---

## 6. 关键 Bug 修复历史

### 6.1 BFN normalize bug

```python
# 错的（早期版本）：
weight_sum = weights.sum()              # 全局 K * T_env ≈ 1600
weights = weights / (weight_sum + 1e-8) # 学习率被除死

# 对的（当前版本）：
weight_per_sa = weights.sum(dim=0, keepdim=True) + 1e-8  # 按 (s,a)
weights = weights / weight_per_sa
```

**影响**：信号放大 3000 倍，BFN 从"没在学"变成"真的在学"。

### 6.2 Φ clamp 防爆炸

| clamp 值 | 效果 |
|----------|------|
| 10 | BFN 输出失控（多次爆炸）|
| 1.0 | log_p 跌，win 跌 |
| **0.3** | **0 爆炸，peak 最高** ⭐ |

### 6.3 logvar clamp

```python
p_logvar = p_logvar.clamp(-2, 2)  # 在 sample_phi 中
```

防止 BFN 内部方差爆炸。

---

## 7. 理论基础

### 7.1 PBRS 策略不变性（Ng et al. 1999）

**定理**：对任意 potential function Φ(s)，塑形奖励 `r' = r + γΦ(s') - Φ(s)` 不改变最优策略。

**直觉**：塑形项是 telescoping sum，在期望 return 中累加为常数 `Φ(s_0)`，不影响 argmax。
**直觉**：塑形项是 telescoping sum，在期望 return 中累加为常数 `Φ(s_0)`，不影响 argmax。

补充一点：MLP 拟合是错的——一个通过 MLP 拟合的奖励函数不可能比原始的、环境给出的奖励函数好。所以我们使用生成式网络（BFN），让它在理论上有探索更优奖励分配的可能性。

考虑过的备选方案（不在本仓库中）：
- **MAVEN**：潜空间分解；离散 code
- **QPLEX**：基于 simplex 的 Q 分解
- **ROMA**：角色感知分解

**对我们意味着什么**：无论 BFN 把 Φ 学成什么样，actor 的最优策略都不变。最坏情况（Φ=0）退化为标准 MAPPO。这是 DRBFN-QVPO 的**安全保证**。

### 7.2 Wiewiora 等价（Wiewiora 2003）

**定理**：在 SARSA 下，PBRS 等价于 Q-value 初始化。

**对我们意味着什么**：用 buffer 里的 a_{t+1}（SARSA 风格）做 PBRS 是合法的，等价于给 Q 一个状态相关的初始偏置。

### 7.3 Q-加权 VLB 的直觉

借鉴 QVPO（NeurIPS 2024）：让生成模型（diffusion / BFN）的 log-likelihood 被 Q 函数的反事实敏感度加权，相当于把生成分布朝"高 Q 梯度方向"偏移。

在我们的设定下，等价于让 BFN 朝"r 与 g_i 对齐"的方向偏移——这样 actor 拿 r 走 PPO 时，自然朝 ∇Q_tot 方向更新，**r 变成了一个学到的、状态相关的优势分解**。

---

## 8. 监控指标

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| `qtot_loss` | Q_tot 训练 loss | 单调下降 |
| `drbfn_loss` | BFN 训练 loss | 负数（log_p 正） |
| `g_n_mean` | Q_tot 输出（团队价值） | 应该增长 |
| `phi_scale` | BFN 输出 Φ 的绝对值平均 | ≤ phi_clamp |
| `g_i_scale` | per-agent Q 敏感度 | 应该增长 |
| `raw_align_std` | K 个 sample 的 align 方差 | 应该增长 |
| `log_p_mean` | BFN log-likelihood | 接近 N × 2.53（max） |
| `grad_norm` | BFN 梯度 | < 20 |

---

## 9. 失败模式识别

| 现象 | 原因 | 解决 |
|------|------|------|
| drbfn_loss ≈ 0 | BFN 没在学 | 检查 normalize（按 (s,a) 不是全局） |
| phi_scale = clamp 上限 | BFN 想输出更大 | OK，已 clamp |
| grad_norm > 100 | 爆炸 | 降低 phi_clamp 或 lr |
| log_p 跌 | BFN samples 离 mean 远 | 监控 align_std |
| g_i_scale 不长 | Q_tot 没学到 per-agent 区分 | 增加 warmup |

---

## 10. 参考文献

- **BFN**: Graves et al. "Bayesian Flow Networks" (2023, [arXiv:2308.07037](https://arxiv.org/abs/2308.07037))
- **QVPO**: Ding et al. "Diffusion-based RL via Q-weighted Variational Policy Optimization" NeurIPS 2024
- **PBRS**: Ng et al. "Policy invariance under reward transformations" (1999)
- **Multi-agent PBRS**: Devlin & Kudenko "Theoretical Considerations of PBRS for Multi-Agent Systems" (2011)
- **Wiewiora 等价**: Wiewiora "Potential-based shaping and Q-value initialization are equivalent" (2003)
- **MAPPO**: Yu et al. NeurIPS 2022
