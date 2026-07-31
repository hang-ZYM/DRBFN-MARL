# DRBFN-QVPO：基于贝叶斯流网络的多智能体信用分配

> 用 **Bayesian Flow Network (BFN)** 学习 per-agent potential 分布 Φ(s, a)，通过 **PBRS（Potential-Based Reward Shaping）** 形式推出细粒度奖励 r。BFN 通过 **Q-加权变分下界（Q-weighted VLB）** 训练——让 BFN 朝高对齐度（r · ∂Q_tot/∂a_i）方向偏移概率密度；actor 拿 r 当 reward 走 PPO，自然朝 ∇Q_tot 方向更新。

**基于**：[marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)（MAPPO 官方实现）
**测试环境**：StarCraft Multi-Agent Challenge (SMAC) — `3m`、`5m_vs_6m`、`2c_vs_64zg`

---

## TL;DR

合作型 MARL 中，团队奖励 `R` 是标量，标准做法给每个智能体分 `R/N`——无法区分贡献。**DRBFN-QVPO** 让 BFN 学一个 per-agent potential `Φ ∈ ℝ^N` 的分布，通过 PBRS 公式得到策略不变的塑形奖励：

```
r_i(t) = R(t)/N + γ·Φ_i(s_{t+1}, a_{t+1}) - Φ_i(s_t, a_t)
```

PBRS 的经典理论（Ng et al. 1999）保证：**任何这种形式的塑形都不改变最优策略**。所以 DRBFN-QVPO 在数学上不会让 actor 学到错误策略——最坏情况退化到 MAPPO。

BFN 通过 Q-加权 VLB 训练，让 Φ 朝"高对齐度"方向偏移——即让 r 与 Q_tot 对动作的反事实敏感度 `g_i = ∂Q_tot/∂a_i` 对齐。这让 actor 拿 r 走 PPO 时，自然朝 ∇Q_tot 方向更新，相当于一个**学到的、状态相关的优势分解**。

---

## 实验结果（截至 2026-07-31）

| Map | Steps | Peak | Final | vs MAPPO baseline |
|---|---|---|---|---|
| `3m` | 1M | **100%** | 96.88% | 持平（简单地图都收敛）|
| **`5m_vs_6m`** | **4.66M** | **90.62%** | 59.38% | **+21.87% over MAPPO 68.75%** |
| `2c_vs_64zg` | 在跑 | TBD | TBD | 已到 34% @ 515K，进度正常 |
| `MMM2` | 2.3M | 0% | 0% | 任务太难（连 MAPPO 都难收敛）|

### `5m_vs_6m` 详细对比（核心结果）

| 算法 | Peak | Mean last 10 |
|---|---|---|
| MAPPO (baseline) | 68.75% | 47.5% |
| **DRBFN-QVPO（本工作）** | **90.62%** | **65.62%** |
| 提升 | **+21.87%** | **+18.12%** |

---

## 核心思想

### 双优化路径

```
Path 2 (lower level, 标准 PPO):
    BFN 生成 Φ → 通过 PBRS 算 r_i → actor 用 r_i 当 reward 更新

Path 1 (upper level, BFN 训练):
    Q_tot 反推 per-agent 敏感度 g_i = ∂Q_tot(s,a)/∂a_i
    BFN 采样 K 组 Φ，每组算 align = r·g
    在 K 个 sample 内归一化，高 align 的样本被加权重
    加权 VLB loss 更新 BFN
```

### 为什么这个设计是新的

| 维度 | DRBFN-QVPO | 传统方法（QMIX/MAVEN/QPLEX）|
|---|---|---|
| 学什么 | per-agent **potential 分布** | per-agent **Q 值点估计** |
| 守恒性 | PBRS 自然满足策略不变性（数学保证）| 显式 value 分解约束 |
| 不确定性 | BFN 后验方差（用于 K-sample argmax 过滤）| 无 |
| 训练信号 | Q-加权 VLB（implicit, end-to-end）| 显式 TD loss |
| 理论基础 | Ng 1999 PBRS + QVPO NeurIPS 2024 + BFN | 各自的 value 分解定理 |

### 关键设计决策

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


### 现在的任务
将三个版本的整合，形成一个完整的算法。

## 仓库结构

```
on-policy/
├── README.md                              # 本文件
├── docs/                                  # 详细文档
│   ├── METHOD.md                          # 算法细节（含 Q-加权 VLB 推导）
│   ├── RESULTS.md                         # 实验结果叙事
│   └── REPRODUCE.md                       # 复现指南
├── tools/                                 # 结果提取与绘图
│   ├── extract_curves.py
│   └── plot_results.py
├── results/                               # 实验产物（已整理）
│   ├── logs/
│   ├── curves/
│   ├── figures/
│   └── tables.md
└── onpolicy/                              # 源代码
    ├── algorithms/
    │   ├── r_drbfn_qvpo/                  # ★ 主推版本（DRBFN-QVPO）
    │   │   ├── README.md                  # 详细算法文档（含代码索引）
    │   │   ├── r_drbfn_qvpo.py            # 主 Trainer
    │   │   └── algorithm/
    │   │       ├── drbfn_qvpo.py          # BFN 模块 (PotentialBFN)
    │   │       └── rDRBFN_QVPOPolicy.py   # Policy 类
    │   └── r_mappo/                       # MAPPO baseline（原始）
    ├── config.py                          # + DRBFN-QVPO 专用参数
    ├── envs/                              # + SMAC 集成
    ├── runner/                            # + DRBFN 训练循环
    └── scripts/                           # + train_smac_qvpo*.sh
```

---

## 安装

**测试通过**：Windows 11 + Python 3.12 + PyTorch 2.x + CUDA 12.x。

```bash
# 1. 克隆
git clone https://github.com/hang-ZYM/DRBFN-MARL.git
cd DRBFN-MARL

# 2. 创建环境
conda env create -f environment.yaml
conda activate marl

# 3. 安装 StarCraft II（SMAC 依赖）
# 从 http://blzdistsc2-a.akamaihd.net/Windows/SC2.4.10.zip 下载 SC2.4.10
# 解压到 ~/StarCraftII/（设置 SC2PATH 环境变量）
# 下载 SMAC 地图：https://github.com/oxwhirl/smac/raw/master/smac_maps/SMAC_Maps.zip
```

完整安装（含 Windows SC2 踩坑）见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

---

## 快速开始

```bash
# 3m（验证 pipeline，1M 步）
python onpolicy/scripts/train/train_smac.py \
    --env_name StarCraft2 \
    --algorithm_name r_drbfn_qvpo \
    --map_name 3m \
    --num_env_steps 1000000 \
    --use_eval --use_linear_lr_decay

# 5m_vs_6m（核心结果，5M 步）
python onpolicy/scripts/train/train_smac.py \
    --env_name StarCraft2 \
    --algorithm_name r_drbfn_qvpo \
    --map_name 5m_vs_6m \
    --num_env_steps 5000000 \
    --use_eval --use_linear_lr_decay

# MAPPO baseline 作对比
python onpolicy/scripts/train/train_smac.py \
    --env_name StarCraft2 \
    --algorithm_name rmappo \
    --map_name 5m_vs_6m \
    --num_env_steps 5000000 \
    --use_eval --use_linear_lr_decay
```

---

## 关键超参数

```bash
# PPO 部分（沿用 MAPPO）
--ppo_epoch 15
--num_mini_batch 1
--clip_param 0.2
--entropy_coef 0.01
--lr 5e-4
--gamma 0.99
--use_linear_lr_decay    # 重要：开启 lr decay

# QVPO 部分
--drbfn_hidden 64                    # BFN 隐藏层
--drbfn_n_sample_steps 2             # BFN 采样步数
--drbfn_lr 3e-4                      # BFN 学习率
--drbfn_warmup_t 20000               # warmup 步数（先用 R/N 训 Q_tot）
--drbfn_K_train 4                    # BFN 训练采样数
--drbfn_K_deploy 4                   # 部署时 K-argmax 数
--drbfn_phi_clamp 0.3                # Φ clamp 范围（关键！）
--drbfn_default_action 0             # default action for g_i（SMAC: no-op=0）
--drbfn_n_step 5                     # n-step return horizon
```

---

## 监控指标

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| `qtot_loss` | Q_tot 训练 loss | 单调下降 |
| `drbfn_loss` | BFN 训练 loss | 负数（log_p 正）|
| `g_n_mean` | Q_tot 输出（团队价值） | 应该增长 |
| `phi_scale` | BFN 输出 Φ 的绝对值平均 | ≤ phi_clamp |
| `g_i_scale` | per-agent Q 敏感度 | 应该增长 |
| `raw_align_std` | K 个 sample 的 align 方差 | 应该增长 |
| `log_p_mean` | BFN log-likelihood | 接近 N × 2.53（max） |
| `grad_norm` | BFN 梯度 | < 20 |

---

## 文档导航

- **[onpolicy/algorithms/r_drbfn_qvpo/README.md](onpolicy/algorithms/r_drbfn_qvpo/README.md)** — 完整算法文档（含代码索引、训练循环详解、关键 bug 修复历史）★ 最详细
- **[docs/METHOD.md](docs/METHOD.md)** — 方法叙事：BFN + PBRS + Q-加权 VLB 推导
- **[docs/RESULTS.md](docs/RESULTS.md)** — 结果分析：5m_vs_6m 90.62% 的解读、失败模式、改进方向
- **[docs/REPRODUCE.md](docs/REPRODUCE.md)** — 复现指南：环境搭建、训练命令、常见陷阱

---

## 引用

```bibtex
@misc{drbfn_qvpo_2026,
  title  = {DRBFN-QVPO: Bayesian Flow Network for Multi-Agent Credit Assignment via Q-Weighted Variational Lower Bound},
  author = {Zhang Yingming},
  year   = {2026},
  url    = {https://github.com/hang-ZYM/DRBFN-MARL}
}
```

参考的原始工作：
- BFN: Graves et al. "Bayesian Flow Networks" (2023)
- QVPO: Ding et al. "Diffusion-based RL via Q-weighted Variational Policy Optimization" NeurIPS 2024
- PBRS: Ng et al. "Policy invariance under reward transformations" (1999)
- Multi-agent PBRS: Devlin & Kudenko (2011)
- Wiewiora 等价: Wiewiora (2003)
- MAPPO: Yu et al. NeurIPS 2022

---

## License

MIT — 见 [LICENSE](LICENSE)。

---

## 状态

**活跃开发中**。2c_vs_64zg 实验进行中；论文撰写中。Issues 和 PR 欢迎。
