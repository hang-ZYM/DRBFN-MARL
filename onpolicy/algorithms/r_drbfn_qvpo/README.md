# DRBFN-QVPO

**Bayesian Flow Network for Multi-Agent Credit Assignment via Q-Weighted Variational Lower Bound**

DRBFN-QVPO 是基于 BFN（Bayesian Flow Network）的多智能体奖励生成算法。BFN 学一个 per-agent potential Φ(s, a) 的分布，r 由 PBRS（Potential-Based Reward Shaping）形式从 Φ 推出。BFN 通过 **Q-加权 VLB** 训练——让 BFN 朝高对齐度（r · ∂Q_tot/∂a_i）的方向偏移概率密度；actor 拿 r 当 reward 走 PPO，自然朝 ∇Q_tot 方向更新。

---

## 1. 核心思想

### 1.1 问题

多智能体协作任务中：
- 环境给团队奖励 R（标量）
- 默认 per-agent 分配（R/N 给每个 agent）**不能区分贡献**
- 真正的 per-agent reward r_i 是隐变量，需要学习

### 1.2 解决方案

让 BFN 学一个生成模型 p_φ(Φ | s, a)，输出 per-agent potential Φ ∈ ℝ^N。通过 PBRS 公式得到 r：

```
r_i(t) = R(t)/N + γ·Φ_i(s_{t+1}, a_{t+1}) - Φ_i(s_t, a_t)
```

BFN 通过 Q-加权 VLB 训练：
```
L_BFN = -Σ_k w^(k) · log p_BFN(Φ^(k) | s, a)
```
其中 w^(k) 由 Q_tot 的反事实敏感度 g_i 决定。

### 1.3 算法骨架（双优化路径）

```
Path 2 (lower level, 标准 PPO):
    BFN 生成 Φ → 通过 PBRS 算 r_i → actor 用 r_i 当 reward 更新

Path 1 (upper level, BFN 训练):
    Q_tot 反推 per-agent 敏感度 g_i = ∂Q_tot(s,a)/∂a_i
    BFN 采样 K 组 Φ，每组算 align = r·g
    在 K 个 sample 内归一化，高 align 的样本被加权重
    加权 VLB loss 更新 BFN
```

---

## 2. 网络结构

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

## 3. 目录结构

```
onpolicy/algorithms/r_drbfn_qvpo/
├── __init__.py
├── r_drbfn_qvpo.py                    # 主 Trainer
└── algorithm/
    ├── __init__.py
    ├── drbfn_qvpo.py                  # BFN 模块 (PotentialBFN)
    └── rDRBFN_QVPOPolicy.py          # Policy 类
```

---

## 4. 文件详解

### 4.1 `algorithm/drbfn_qvpo.py` — BFN 模块

#### `CtsBayesianFlow`
BFN 的前向（加噪）过程，参数化为：
```
post_var = min_variance^t
alpha_t = 1 - post_var
mean = alpha_t * x_0 + sqrt(alpha_t * post_var) * noise
```

关键方法：
- `forward(x_0, t)`: 给数据 x_0 加噪到时间 t
- `get_alpha(i, n_steps)`: 第 i 步的精度参数
- `update_input_params(params, y, alpha)`: Bayesian 更新

#### `ConditionNet`
条件去噪网络：`(x_t, t, h) → (mean, logvar)`

#### `PotentialBFN`（核心）
BFN 主体，输入 `(s, a_joint)`，输出 Φ ∈ ℝ^N。

**关键方法**：

```python
encode(raw):           # (s, a) → 隐藏向量 h
sample_phi(h):         # 随机采样 Φ（用于部署和训练 K 个 sample）
log_prob_phi(phi, h):  # 计算 log p_BFN(Φ)（用于 VLB 训练）
sample(raw):           # 接口：raw → Φ
```

**`sample_phi`**：BFN 反向采样过程
```python
for i in 1..n_sample_steps:
    out = g_net(mean, t, h)
    p_mean, p_logvar = out
    output_sample = p_mean + p_std * randn   # 随机
    y = sender_dist(output_sample, alpha).sample()  # 随机
    update mean using y
return g_net(mean, t_final, h)[:, :D]
```

**`log_prob_phi`**：用**固定 noise_dev = sqrt(min_variance)** 算 Gaussian log-prob
```python
# 关键：用固定 noise_dev，不用网络预测的 logvar
# 原因：防止 σ 塌缩到 0
noise_dev = sqrt(min_variance) = sqrt(1e-3) ≈ 0.032
log_p = -0.5 * (phi - pred_mean)^2 / var - 0.5*log(2π·var)
```

### 4.2 `algorithm/rDRBFN_QVPOPolicy.py` — Policy

继承 `R_MAPPOPolicy`（获得 actor + V-critic），添加：
- `drbfn = PotentialBFN(...)`: BFN 网络
- `qtot_net = GlobalQ(...)`: Q_tot 网络
- `qtot_target`: Q_tot 的 target 网络
- `drbfn_optimizer`, `qtot_optimizer`: 优化器
- `soft_update_qtot()`: soft update target
- `lr_decay(episode, episodes)`: **对所有 4 个网络做 lr decay**

### 4.3 `r_drbfn_qvpo.py` — Trainer

继承 `R_MAPPO`，核心方法：

#### `train(buffer, update_actor=True)`
主训练循环：
1. 判断是否在 warmup
2. **算 per-agent r**（warmup 时用 R/N，否则用 PBRS 公式）
3. 把 r 替换到 buffer
4. 跑 PPO（lower level）
5. 恢复 buffer
6. 训 Q_tot（n-step return on R）
7. 训 BFN（Q-加权 VLB）

#### `_compute_warmup_rewards(buffer)`
warmup 期间的 r：直接返回 R/N。

#### `_compute_pbrs_rewards(buffer)` ⭐ 核心
部署时的 r 计算（K-sample argmax）：
```python
for each (s, a) in buffer:
    g_i = compute_g_i(s, a)              # per-agent Q 敏感度
    for k in 1..K_deploy:
        phi_k = sample_phi(h).clamp(-phi_clamp, phi_clamp)
        phi_next_k = sample_phi(h_next).clamp(...)
        r_k = R/N + γ·phi_next_k·mask - phi_k
        align_k = (r_k * g_i).sum()
    r = r^(argmax_k align_k)             # 选 align 最大的
return r
```

#### `_compute_g_i(so_flat, joint_a_flat)` ⭐ 核心
default action 反事实：
```python
q_actual = Q_tot(s, a)
for i in 1..N:
    a_default_i = joint_a.clone(); a_default_i[:, i, :] = 0; a_default_i[:, i, default_action] = 1
    q_default_i = Q_tot(s, a_default_i)
    g_i = q_actual - q_default_i
```

#### `_train_qtot(buffer)`
n-step return 训 Q_tot：
```
G_n = Σ_{k=0}^{n-1} γ^k R_{t+k} + γ^n Q_target(s_{t+n}, a_{t+n})
loss = MSE(Q_tot(s, a), G_n)
```

#### `_train_bfn(buffer)` ⭐ 核心
Q-加权 VLB 训练 BFN：
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
weights = F.relu(aligns)
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

## 6. 关键修复（重要 Bug）

### 6.1 BFN normalize bug（v1 → v2）

```python
# 错的（v1）：
weight_sum = weights.sum()              # 全局 K * T_env ≈ 1600
weights = weights / (weight_sum + 1e-8) # 学习率被除死

# 对的（v2）：
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

## 7. 配置参数

### 7.1 PPO 部分（沿用 MAPPO）
```bash
--ppo_epoch 15
--num_mini_batch 1
--clip_param 0.2
--entropy_coef 0.01
--lr 5e-4
--gamma 0.99
--use_linear_lr_decay    # 重要：开启 lr decay
```

### 7.2 QVPO 部分
```bash
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

## 8. 训练循环详解

```
for iteration = 1, 2, ...:
    
    # === Phase 1: Rollout（用当前 actor + BFN）===
    for t = 1 to T:
        sample a_i ~ π_i(·|o_i)
        execute joint a, observe R, s'
        
        # K-sample argmax 选 r
        if iteration > warmup:
            Φ_star = K_argmax(BFN, s, a, Q_tot, K=K_deploy)
        else:
            Φ_star = 0  # 默认 R/N
        
        # 算 r via PBRS（SARSA 风格 a'）
        if t < T:
            r_i = R/N + γ Φ_star_i(s', a') - Φ_star_i(s, a)
        else:
            r_i = R/N - Φ_star_i(s, a)  # terminal
        
        store (s, a, R, s', Φ_star, r) in buffer
    
    # === Phase 2: Lower level (PPO) ===
    A_i = r_i + γ Q_tot(s', a') - Q_tot(s, a)  # Q_tot 当 critic
    PPO update actor + V-critic
    
    # === Phase 3: Q_tot update (on R, not r) ===
    G_n = Σ_{k=0}^{n-1} γ^k R_{t+k} + γ^n Q_tot_target(s_{t+n}, a_{t+n})
    minimize MSE(Q_tot(s, a), G_n)
    soft update Q_tot_target
    
    # === Phase 4: BFN update (Q-weighted VLB) ===
    if iteration > warmup:
        g_i = Q_tot(s, a_i, a_{-i}) - Q_tot(s, default_i, a_{-i})  # 反事实
        for k = 1..K_train:
            Φ^(k) = sample_phi(h)
            align^(k) = (r^(k) · g_i).sum()
            log_p^(k) = log_prob_phi(Φ^(k), h)
        
        aligns = K_normalize(aligns)
        weights = ReLU(aligns) / sum_per_sa(weights)
        
        L_BFN = -Σ_k weights^(k) · log_p^(k)
        update BFN
```

---

## 9. 如何运行

### 9.1 SMAC 训练

```bash
# 3m
python onpolicy/scripts/train/train_smac.py \
    --env_name StarCraft2 \
    --algorithm_name r_drbfn_qvpo \
    --map_name 3m \
    --num_env_steps 1000000 \
    --use_eval --use_linear_lr_decay \
    --use_wandb \
    [其他参数...]

# 5m_vs_6m（推荐）
python onpolicy/scripts/train/train_smac.py \
    --env_name StarCraft2 \
    --algorithm_name r_drbfn_qvpo \
    --map_name 5m_vs_6m \
    --num_env_steps 5000000 \
    --use_eval --use_linear_lr_decay \
    --use_wandb \
    [其他参数...]

# 2c_vs_64zg
python onpolicy/scripts/train/train_smac.py \
    --env_name StarCraft2 \
    --algorithm_name r_drbfn_qvpo \
    --map_name 2c_vs_64zg \
    --num_env_steps 5000000 \
    --use_eval --use_linear_lr_decay \
    --use_wandb \
    [其他参数...]
```

### 9.2 toy 测试

```bash
cd on-policy
python test_toy_qvpo.py --n_agents 3 --episodes 50
```

---

## 10. 监控指标

### 10.1 关键指标（每个 BFN step 打印）

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

### 10.2 失败模式识别

| 现象 | 原因 |
|------|------|
| drbfn_loss ≈ 0 | BFN 没在学（检查 normalize）|
| phi_scale = clamp 上限 | BFN 想输出更大（OK，已 clamp）|
| grad_norm > 100 | 爆炸（降低 phi_clamp 或 lr）|
| log_p 跌 | BFN samples 离 mean 远 |
| g_i_scale 不长 | Q_tot 没学到 per-agent 区分 |

---

## 11. 实验结果（截至 2026-07-31）

### 11.1 已完成实验

| Map | Steps | Peak | Final | 状态 |
|-----|-------|------|-------|------|
| **3m** | 1M | **100%** | 96.88% | 18 次爆炸但能恢复 |
| **5m_vs_6m** | 4.66M | **90.62%** | 59.38% | 后期轻微不稳定 |
| **2c_vs_64zg** | 在跑 | TBD | TBD | 已到 34% @ 515K |
| MMM2 | 2.3M | 0% | 0% | 任务太难 |

### 11.2 vs baselines（5m_vs_6m）

| 算法 | Peak | Mean last 10 |
|------|------|--------------|
| MAPPO | 68.75% | 47.5% |
| v1 (DRBFN) | 75% | 60.9% |
| v_final | 62.5% | 43.9% |
| **QVPO** | **90.62%** ⭐ | **65.62%** |

---

## 12. 后续工作

### 12.1 已知问题
1. **后期不稳定**（5m_vs_6m update 1000+ 后 log_p 下降）
2. **MMM2 学不会**（任务太难 + 死循环）
3. **SC2 长跑稳定性**（多次 SC2 崩溃中断）

### 12.2 改进方向
1. **BFN target network**（slow-moving BFN，防分布漂移）
2. **更激进的 entropy reg**（避免 actor 过早收敛）
3. **Curiosity-driven exploration**（解决 MMM2 死循环）
4. **lr decay 已加入**（验证对后期稳定性的影响）

### 12.3 实验计划
- [ ] 2c_vs_64zg 完整 5M
- [ ] 6h_vs_8z（hard 同质 map）
- [ ] corridor（super hard）
- [ ] 多 seed 验证

---

## 13. 参考

- **BFN 原论文**: Graves et al. "Bayesian Flow Networks" (2023)
- **QVPO**: Ding et al. "Diffusion-based RL via Q-weighted Variational Policy Optimization" NeurIPS 2024
- **PBRS**: Ng et al. "Policy invariance under reward transformations" (1999)
- **Multi-agent PBRS**: Devlin & Kudenko "Theoretical Considerations of PBRS for Multi-Agent Systems" (2011)
- **Wiewiora 等价**: Wiewiora "Potential-based shaping and Q-value initialization are equivalent" (2003)
- **TAR²**: Kapoor et al. "Redistributing Rewards Across Time and Agents for MARL" (2025)
- **Yang 2024**: Yang et al. "Learning Individual Potential-Based Rewards in MARL" (2024)
