# DRPO Design Doc：把 DRBFN 奖励分解迁移到 GRPO 的可行性分析

> 探索将 MARL 中的 BFN 奖励分解思路应用到 LLM 推理 RL 的 step-level credit assignment。本文档**不粉饰**：先讲清楚相关工作有多少，再分析真正新颖的空间在哪。

---

## 0. TL;DR（先说结论）

**核心判断**：DRPO 的 high-level idea（"把最终 reward 分解到 reasoning chain 的每一步"）**已经被多个工作探索过**，且 2025-2026 年非常密集（VinePPO, OAR, OPPO, PBSD, FlowTracer, PURE 等）。

**真正新颖的空间**在于：
1. **没人用 BFN**（Bayesian Flow Network）作为分解器——这是个理论扎实但 LLM 圈不熟悉的工具
2. **没人系统利用 BFN 的不确定性**作为安全网（Dual Gate）——这点在 reward hacking 普遍的 LLM RL 里有实用价值
3. **MARL → LLM 的迁移视角**本身在 LLM RL 论文里很罕见

**可行性结论**：
- ✅ 理论基础扎实（BFN 有概率推断等价性）
- ✅ 工程上可复用 90% 现有 DRBFN 代码
- ⚠️ 算法新颖性中等（需要靠"BFN 视角 + 不确定性安全网"打差异）
- ⚠️ 实验压力较大（需要超越 VinePPO / OAR 等 strong baseline）
- 🎯 推荐定位为"应用论文"（application paper）而非纯算法论文——把 MARL 经验系统迁移到 LLM RL

---

## 1. 问题陈述

### 1.1 GRPO 的 credit assignment 痛点

GRPO 在长 CoT 任务上有一个根本问题：**最终答案的 scalar reward 被均匀分配给所有 token**。

```
Prompt → 1500 token CoT → <answer>42</answer>  ← reward = 0 (wrong)

GRPO 处理：
  对整个 1500 token 序列打 0 分
  → 所有 token 的 advantage = -mean(group) / std
  → 没有 token-level 信号告诉模型"哪一步开始错的"
  → 长推理链的信用分配严重稀释
```

引用 **VinePPO (Kazemnejad et al., ICML 2025)** 的实验：标准 PPO 的 value network 在这种长链任务上"barely outperform a random baseline when comparing alternative steps"。

### 1.2 为什么这是问题

R1-Zero 风格训练的 hallmark 就是**自发涌现长链推理**（>1000 tokens）。但 DAPO 的 token-level loss normalization 只解决了"长序列被隐式惩罚"的问题，**没有解决"哪步该被奖励"的信用分配问题**。

具体表现：
- 模型生成 1500 token 推理，最后答案错
- 模型不知道是第 3 步算错了，还是第 12 步误导了
- 只能整体惩罚，导致 learning signal 极稀疏
- 训练效率低，需要海量样本才能学会"避免某些错误模式"

---

## 2. 相关工作全景（重要）

### 2.1 Step-level Credit Assignment in LLM RL（直接相关，最密集）

| 工作 | 会议/时间 | 核心方法 | 与 DRPO 的关系 |
|---|---|---|---|
| **VinePPO** | ICML 2025 | MC rollout per step，无 critic | 直接竞品，最干净的 step-level 方法 |
| **OAR** | ACL 2026 | counterfactual token perturbation + gradient approximation | token-level 而非 step-level；OAR-G 单次 backward 高效 |
| **Counterfactual Credit Assignment** | OpenReview 2026 (arxiv 2602.09331) | **mask reasoning spans，measure answer probability drop** | **几乎和 DRPO high-level idea 一样！** |
| **OPPO** | arxiv 2605.21851 (May 2026) | **Bayesian** value recursion for token-level | 已经在用 Bayesian 框架，但不是 BFN |
| **PBSD** | arxiv 2606.09348 (June 2026) | **Bayesian** self-distillation，teacher-student likelihood ratio | 也是 Bayesian，但用 privileged info（ground truth） |
| **IBPO** | arxiv 2605.16302 (April 2026) | implicit process-level signals from trajectory comparison | 不需 step supervision |
| **FlowTracer** | arxiv 2606.10646 (June 2026) | attention graph + max-flow 找推理骨干 | 完全不同的视角（attention flow）|
| **Probabilistic Flow Reasoning** | ACL 2026 (arxiv 2601.09260) | Rectified Flow 学 dense reward transport | 用 flow 概念但不是 BFN |
| **PURE (Min-form PRM)** | NeurIPS 2025 | 解决 PRM 的 reward hacking；min-form 而非 sum-form | 解决一个 DRPO 也要面对的问题 |
| **Q-RM** | 2025 | token-level Q-value as reward | 另一种 token-level 方法 |

### 2.2 Process Reward Models（间接相关）

| 工作 | 核心方法 | 局限 |
|---|---|---|
| **OpenAI PRM800K** (Lightman et al., 2024) | 人工标注 step-level reward | 极贵，~800k 标注 |
| **R-PRM** (EMNLP 2025) | reasoning-driven PRM with self-evolution | 需要 cold start 数据 |
| **CAPO** | generative credit assignment | 依赖 LLM-as-judge |

### 2.3 GRPO 变体（基础）

- **GRPO** (DeepSeekMath, 2024)：组相对优势，省 critic
- **DAPO** (ByteDance, NeurIPS 2025)：4 项技术（Clip-Higher / Dynamic Sampling / **Token-level loss** / Overlong Reward Shaping）
- **Dr. GRPO**：简化版

### 2.4 MARL 中的奖励分解（你的领域）

- **QMIX / QPLEX / MAVEN**：value decomposition
- **ROMA**：role-aware decomposition
- **DRBFN（你的工作）**：BFN 做随机分解，Dual Gate 安全网，n-step counterfactual

### 2.5 关键观察

**两条线还没真正交叉**：
- LLM RL 圈的 step-level credit 工作几乎都是"用 LLM 自己估计"（attention, MC rollout, counterfactual token）
- MARL 圈的奖励分解工具（BFN, stick-breaking, Bayesian）**几乎没出现在 LLM RL 论文里**

这就是 DRPO 的真正空间：**把 MARL 的成熟奖励分解工具带到 LLM RL**。

---

## 3. DRPO 设计

### 3.1 核心思想（一句话）

把 DRBFN 的 BFN 奖励分解器从"多 agent 之间"重定向到"长 CoT 的多步之间"，给每个 reasoning step 学一个细粒度奖励，再喂给 GRPO 训练 LLM。

### 3.2 架构总览

```
┌────────────────────────────────────────────────────────────────┐
│                    一个 DRPO 训练步                            │
└────────────────────────────────────────────────────────────────┘

  prompt ──► vLLM 生成 G=8 completions（同 GRPO）
                  │
                  ▼
              每条 completion:
              ┌─────────────────────────────────────┐
              │  1. 计算 final reward R              │
              │     (correctness + format, 同 GRPO) │
              │  2. CoT 分段 → K steps               │
              │  3. BFN 分解 R → (r_1, ..., r_K)    │
              │  4. Dual Gate: gated_r_i             │
              └─────────────────────────────────────┘
                  │
                  ▼
              GRPO 优势计算（用 step-level reward 替代 scalar）
                  │
                  ▼
              DAPO loss + actor update
                  │
                  ▼
              周期性 BFN 重训（counterfactual rollout）
```

### 3.3 与 DRBFN 三版本的对应映射

| DRBFN（MARL）| DRPO（LLM）| 实现差异 |
|---|---|---|
| N agents | K reasoning steps | "agent" 重新定义为 CoT step |
| 全局 reward R | 最终答案 correctness | 团队奖励 = 答案正确性 |
| Per-agent reward r_i | Per-step reward r_i | step-level 信用 |
| Stick-breaking 硬守恒（v1）| 软守恒（推荐 v2 范式）| 允许"协同步骤"，因 step 之间有非线性交互 |
| Counterfactual rollout (v3) | **Step replace + 续写** | 替换某步后用 reference policy 续写 |
| Dual Gate MAPPO 回退 | Dual Gate 均匀回退 (R/K) | BFN 不确定时退到原始 GRPO 行为 |
| Bellman residual 训练 | Step-level Q 训练 | 用 step-level Q_i 学 |

### 3.4 BFN 分解器的具体架构

```python
class DRPO_BFN(nn.Module):
    """CoT-level BFN: 把 R 分解到 K 个 reasoning steps。"""

    def __init__(self, llm_hidden_dim, hidden=512, K_max=20):
        super().__init__()
        self.encoder = LLMEncoder(frozen=True)  # 用 actor 的 hidden states
        self.step_encoder = nn.Sequential(
            nn.Linear(llm_hidden_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        # BFN components (复用 DRBFN v1/v2 代码)
        self.bayesian_flow = CtsBayesianFlow(min_variance=1e-3)
        self.condition_net = ConditionNet(cond_dim=hidden, n_steps_max=K_max)
        # Stick-breaking head (K_max-1 weights)
        self.stick_head = nn.Linear(hidden, K_max - 1)

    def forward(self, prompt_hidden, step_hiddens, R, K):
        """Returns: (r_1, ..., r_K), Var[r_i]."""
        # 1. 编码每个 step
        step_embs = self.step_encoder(step_hiddens)  # (B, K, hidden)
        # 2. 聚合全局 context
        context = (step_embs * step_attn_mask).mean(dim=1) + prompt_hidden
        # 3. BFN 生成（K-1 个 stick-breaking weights）
        weights = self.bayesian_flow.sample(...)  # 见 DRBFN v1
        # 4. Stick-breaking conservation: Σ r_i = R
        rewards = stick_breaking_decompose(weights, R, K)
        # 5. 不确定性
        variances = self.bayesian_flow.posterior_variance(...)
        return rewards, variances
```

**关键复用**：`CtsBayesianFlow` / `ConditionNet` / stick-breaking 逻辑全部来自 `onpolicy/algorithms/utils/drbfn.py`，几乎不改。

### 3.5 Dual Gate（防 reward hacking）

```python
def dual_gate(r_bfn, var_bfn, R, K):
    """BFN 不确定时退到均匀分解（即原始 GRPO 行为）。"""
    confidence = 1.0 / (1.0 + var_bfn)  # (B, K)
    uniform = R / K  # scalar broadcast
    gated = confidence * r_bfn + (1 - confidence) * uniform
    return gated
```

**为什么这个重要**：LLM RL 最常见的失败模式就是 reward hacking——BFN 的方差提供天然防御。

### 3.6 训练循环（详细伪代码）

```python
# Phase 1: Warmup (前 N_warmup 步)
# - 冻结 BFN
# - 用 R/K 均匀分解（即原始 GRPO）
# - 让 actor 学到合理 baseline

# Phase 2: BFN 训练 (周期性，每 N_bfn 步)
def train_bfn_step(actor, bfn, buffer):
    for sample in buffer.sample(100):
        completion = sample.completion
        K = len(segment(completion))
        for i in range(K):
            # Counterfactual rollout
            alt_step = reference_policy.generate(prefix=completion[:i])
            alt_completion = alt_step + actor.generate(prefix=...)
            R_cf = reward_fn(alt_completion)
            delta_i = sample.R - R_cf  # 真实贡献
            # BFN 预测
            r_pred, var = bfn(completion[:i+1], sample.R)
            # Loss (Bellman residual)
            loss_bfn += (r_pred[i] - delta_i)**2 + torch.log(var[i])
        loss_bfn.backward()

# Phase 3: 联合训练 (主循环)
def drpo_step(actor, bfn, optim_actor, optim_bfn, batch):
    completions = vllm_generate(actor, batch, G=8)
    R = reward_fn(completions)
    # BFN 分解
    steps = segment(completions)  # K steps per completion
    r_bfn, var_bfn = bfn(prompt_hidden, step_hiddens, R, K)
    # Dual gate
    gated_r = dual_gate(r_bfn, var_bfn, R, K)
    # GRPO 优势（用 step-level reward）
    advantages = group_advantage(gated_r)  # 同 GRPO 但 reward 是 step-level
    # DAPO loss
    loss_actor = dapo_loss(actor, completions, advantages)
    optim_actor.step(loss_actor)
    # 周期性更新 BFN（不是每步）
    if step_count % 50 == 0:
        train_bfn_step(actor, bfn, buffer)
```

---

## 4. 理论基础

### 4.1 BFN 的概率推断等价性

BFN（Bayesian Flow Network, Gao et al. 2023）在数学上等价于一个**迭代贝叶斯推断**过程：给定观测（这里：completion 的 hidden states + final reward R），逐步更新对 per-step reward 的后验分布。

形式化：
- Prior: `p(r_1, ..., r_K)` = uniform（无信息）
- Likelihood: `p(R | r_1, ..., r_K) = δ(Σr_i = R)` (守恒)
- Posterior: `p(r_1, ..., r_K | R, context)` ≈ BFN 输出

这给 DRPO 一个**理论锚点**：BFN 不是任意神经网络，而是**变分推断**的一个具体实例。

### 4.2 守恒的结构性保证

**Stick-breaking parameterization**（v1）确保：

```
r_1 = w_1 · R
r_2 = (1 - w_1) · w_2 · R
...
r_K = Π_{j<K} (1 - w_j) · R

Σ r_i = R  (telescoping sum, by construction)
```

这给 DRPO 一个**硬约束保证**：分解后的奖励之和精确等于最终奖励。这在 LLM RL 里很重要——避免"BFN 学会放大 reward"导致训练不稳定。

### 4.3 Dual Gate 的 PAC-Bayes 解释

Dual Gate 的"BFN 不确定时回退到均匀"对应一个 PAC-Bayes 风格的鲁棒性保证：在 worst-case 下（BFN 完全错），DRPO 退化为标准 GRPO，因此**DRPO 在理论上不会比 GRPO 差**（modulo warmup 期间的 BFN 训练开销）。

形式化：设 GRPO 的 expected return 为 `J_GRPO`，DRPO 为 `J_DRPO`，则

```
J_DRPO ≥ J_GRPO - O(warmup_cost)
```

这是 DRPO 安全性的核心保证。

### 4.4 Counterfactual Target 的无偏性

借鉴 **VinePPO** 的关键 insight：在 LLM 推理里，**counterfactual rollout 可以精确计算**（不像传统 RL 需要近似）。把 step i 替换后重新续写，得到 R_cf_i，那么 Δ_i = R - R_cf_i 是 step i 真实因果效应的**无偏估计**。

DRPO 用 BFN 学习预测这个 Δ_i，等价于学习一个 amortized 版本的 VinePPO——**训练时贵一次，推理时只用一次前向传播**。

---

## 5. 与现有工作的差异化分析（关键）

### 5.1 vs VinePPO（最强竞品）

| 维度 | VinePPO | DRPO |
|---|---|---|
| Step value 估计 | MC rollout per step（每步几十次续写）| BFN 单次前向 |
| 训练成本 | 高（每步 K 倍推理）| 中（BFN 训练贵，推理便宜）|
| 不确定性 | 无 | BFN 后验方差 + Dual Gate |
| 守恒性 | 无（V(s) 不强制守恒）| Stick-breaking 硬守恒 |
| Memory | 不需要额外网络 | +5M params BFN |

**DRPO 的优势**：推理时便宜（VinePPO 推理时还要做 MC）；有不确定性安全网
**DRPO 的劣势**：需要 warmup + 周期 BFN 重训；BFN 可能学不好

### 5.2 vs Counterfactual Credit Assignment (arxiv 2602.09331)

**这是最像的工作**——也是 mask spans + measure answer probability drop。

| 维度 | 他们的方法 | DRPO |
|---|---|---|
| 信用来源 | policy model 自己的概率变化 | 独立的 BFN 分解器 |
| Span 检测 | 模式匹配（算式、句子边界）| 同（CoT 分段）|
| 推理成本 | mask + 多次 forward | 一次 BFN forward |
| 标注需求 | 无 | 无（counterfactual rollout 自监督）|

**DRPO 的差异化**：BFN 提供 uncertainty；可以学比"mask 算式"更细粒度的 credit（句子级 vs token 级）

### 5.3 vs OPPO（Bayesian Value Recursion）

OPPO 用 Bayesian update rule 估计 token-level credit。

| 维度 | OPPO | DRPO |
|---|---|---|
| Bayesian 工具 | 简单的递归更新 | BFN（变分推断）|
| Token-level vs step-level | Token | Step（粒度更粗但语义更清晰）|
| Critic-free | ✅ | ✅ |
| 不确定性 | 隐式（V_t 变化）| 显式（BFN variance）|

**DRPO 的差异化**：BFN 是更通用的工具；step-level 更适合 R1-Zero 风格的长 CoT

### 5.4 vs PBSD（Privileged Bayesian Self-Distillation）

PBSD 用 ground-truth answer 作为 privileged info 比较 likelihood。

**DRPO 的差异化**：DRPO 不需要 ground truth（只用 R）；PBSD 的 teacher-student 框架更重

### 5.5 DRPO 真正的独特性（诚实评估）

**真正独特的只有两点**：
1. **BFN 作为分解器**——LLM RL 圈几乎没人用，但 BFN 有变分推断理论支撑
2. **Dual Gate 作为 reward hacking 防御**——直接对接 PURE/Min-form PRM 想解决的问题，但用 BFN uncertainty 实现而不是改 loss form

**不独特的部分**：counterfactual rollout（VinePPO 已做）、step-level reward（很多工作在做）、Bayesian framework（OPPO/PBSD 已做）

---

## 6. 可行性分析

### 6.1 工程可行性（高）

- ✅ 90% DRBFN 代码可直接复用（BFN 网络、Dual Gate、训练循环框架）
- ✅ 90% tinyzero_grpo 代码可直接复用（GRPO trainer、vLLM colocate、rewards）
- ✅ 单卡 MI300X 够用（7B actor + 5M BFN + vLLM colocate，预算充足）
- ⚠️ Counterfactual rollout 需要小心：要么用 reference policy，要么用 vLLM 单独跑

### 6.2 算法可行性（中）

**风险点**：
1. **CoT 分段不稳定**：countdown 算式边界清晰，GSM8K 是自然语言句子，分段策略影响很大
   - 缓解：先在 countdown 上验证，再扩到 GSM8K
2. **BFN 学习难度**：counterfactual rollout 信号可能很 noisy
   - 缓解：Dual Gate 保证退化到 GRPO，不会更差
3. **Reward hacking**：BFN 本身可能被 hack
   - 缓解：Dual Gate + 周期性 ground truth 校准

### 6.3 论文新颖性（中等偏低，需要慎重 framing）

**坦诚讲**：
- High-level idea（step-level credit assignment）已经被探索得很彻底
- BFN 视角是新的，但 reviewer 可能问"为什么不用更简单的 OPPO？"
- 真正的卖点是 **"MARL → LLM RL 的系统迁移"** 而非"新算法"

**建议的论文 framing**：
- ❌ 不要写"我们提出 DRPO，一个新算法"
- ✅ 写"我们系统研究如何将 MARL 的奖励分解工具迁移到 LLM RL，发现 BFN 提供 LLM 圈不熟悉的 uncertainty-aware 安全网"

### 6.4 实验可行性（中-高）

**核心实验**（必须做）：
1. Countdown（你已有 setup）: DRPO vs GRPO baseline
2. GSM8K（更长 CoT）: DRPO vs GRPO baseline

**Strong baseline 对比**（要做）：
- VinePPO（implementation 公开）
- OAR-G（梯度近似，单次 backward，工程友好）
- OPPO（Bayesian recursion，无需训练）

**目标会议**：ICLR / NeurIPS workshop / ACL（reasoning track）

---

## 7. 实验设计（详细）

### 7.1 Setup

- 模型：Qwen2.5-7B-Instruct
- 任务：Countdown（已有数据），GSM8K（标准数学推理）
- 硬件：单卡 MI300X
- 时长预算：每个方法 1200 step × 3 seeds

### 7.2 Baselines

| Baseline | 实现成本 | 必要性 |
|---|---|---|
| GRPO (R1-Zero) | 0（你已有）| ★★★★★ 必做 |
| GRPO + DAPO token-norm | 0（你已有）| ★★★★★ 必做 |
| VinePPO | 中（公开 repo）| ★★★★ 强烈推荐 |
| OAR-G (gradient-based) | 中（ACL 2026 paper）| ★★★ 推荐 |
| OPPO | 低（简单递归）| ★★ 可选 |

### 7.3 Metrics

| 指标 | 用途 |
|---|---|
| **Correctness** | 核心性能 |
| **Sample efficiency** | 收敛到 X% correctness 所需步数 |
| **Mean CoT length** | 是否帮助模型更高效推理（避免 overthinking）|
| **Step-level credit accuracy** | BFN 预测的 r_i 与 counterfactual ground truth 的相关性 |
| **Reward hacking robustness** | 人为注入"作弊模板"，看 Dual Gate 是否拦截 |
| **Ablation: BFN off vs on** | 量化 BFN 分解的实际增益 |
| **Ablation: Dual Gate off vs on** | 量化 Dual Gate 的安全价值 |

### 7.4 假设（要预测的）

| H# | 假设 | 信心 |
|---|---|---|
| H1 | DRPO 在 Countdown 上略优于 GRPO（任务简单，价值有限）| 中 |
| H2 | DRPO 在 GSM8K 上显著优于 GRPO（长 CoT 价值大）| 高 |
| H3 | DRPO 收敛比 GRPO 快（步级信号更密）| 中 |
| H4 | Dual Gate 在 reward hacking 注入实验中显著优于纯 BFN | 高 |
| H5 | DRPO 不如 VinePPO（VinePPO 的 MC 是无偏的）| 中 |

**如果 H5 成立**（DRPO 不如 VinePPO），论文卖点是"sample-efficient 推理 + uncertainty 安全网"。

---

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| BFN 学不好（counterfactual 信号太 noisy）| 高 | 高 | Dual Gate 保证退化到 GRPO |
| CoT 分段策略影响大 | 中 | 中 | 多种 split 对比；先用 countdown 验证 |
| Counterfactual rollout 太贵 | 高 | 中 | 只对部分样本做（10%）；用小模型续写 |
| 被 reviewer 说"和 X 一样" | 高 | 高 | 诚实 related work；强调 BFN 不确定性 + MARL 视角 |
| 7B 上跑不下 BFN + actor | 低 | 高 | BFN 只有 5M 参数，可放到 CPU offload |
| Dual Gate 一直触发（BFN 一直不确定）| 中 | 中 | 监控 active_ratio；增加 warmup |

---

## 9. 时间表（如果决定做）

| 阶段 | 时长 | 内容 |
|---|---|---|
| Week 1 | 7 天 | 实现 CoT 分段 + BFN 分解器（基于现有 DRBFN 代码）|
| Week 2 | 7 天 | 实现 counterfactual rollout + BFN 训练循环 |
| Week 3 | 7 天 | Countdown 上的 smoke test + 调试 |
| Week 4-5 | 14 天 | Countdown 完整实验（GRPO baseline + DRPO，3 seeds）|
| Week 6-7 | 14 天 | GSM8K 实验（核心结果）|
| Week 8 | 7 天 | VinePPO / OAR-G baseline 对比 |
| Week 9-10 | 14 天 | Ablation + 写 paper |

**总计**：~10 周（2.5 个月）。这与你的求职 timeline（提前批已经开闸）冲突，建议作为入职后的第一个项目，而不是求职前的研究。

---

## 10. 建议（务实判断）

### 10.1 不建议现在做 DRPO（求职前）

理由：
1. **秋招提前批已经开闸**——你的优先级应该是投简历 + 面试
2. **2.5 个月 timeline** 会错过秋招窗口
3. **算法新颖性中等**——不是稳赚 paper 的方向
4. **风险高**——VinePPO / OAR 等竞品已经做得很好，超越有难度

### 10.2 建议作为入职后第一个项目

理由：
1. **大厂资源**（字节 Seed / DeepSeek / 阿里通义）有更多 GPU 和 mentor
2. **DRPO 是个完整的研究项目**，能产 1 篇 paper + 多个 ablation
3. **结合你的 MARL 背景**，独特性强
4. **在面试时讲**：作为"我的研究规划"，体现深度

### 10.3 短期（求职前）应该做什么

把 DRPO 作为**面试时的研究规划**讲，而不是实际项目：

> "我的研究方向是把 MARL 里的奖励分解工具系统迁移到 LLM RL。我在 DRBFN 项目里用 BFN 做多 agent 奖励分解，发现这套工具正好可以解决 LLM RL 里的 step-level credit assignment 问题。我已经详细调研了 VinePPO / OAR / OPPO 等相关工作，发现 BFN 的不确定性量化是一个 LLM RL 圈还没充分探索的角度。我计划在大模型岗入职后系统研究这个方向。"

这个回答会让面试官觉得你：
- 有研究 sense（系统调研相关工作）
- 有差异化（BFN 视角独特）
- 有判断力（不当盲目算法新颖性主义者）
- 有规划（不是临时想的）

---

## 11. 这份文档的归宿

建议保存到：
- DRBFN repo: `docs/DRPO_design.md`（作为 DRBFN 的延伸研究方向）
- 简历附录（如果面试时需要研究规划）
- 个人 Obsidian vault（长期参考）

**不要作为独立 GitHub repo 上传**——避免给 reviewer 留下"想法已经公开"的印象（影响将来投稿）。

---

## 12. 参考文献（按时间倒序）

### LLM RL Step-level Credit Assignment

1. FlowTracer (arxiv 2606.10646, 2026-06) — Attention DAG + max-flow
2. PBSD (arxiv 2606.09348, 2026-06) — Bayesian self-distillation
3. OPPO (arxiv 2605.21851, 2026-05) — Bayesian value recursion
4. IBPO (arxiv 2605.16302, 2026-04) — Implicit process signals
5. Counterfactual Credit Assignment (arxiv 2602.09331, 2026-02) — Mask spans + probability drop
6. OAR (ACL 2026, arxiv 2601.09260) — Outcome-grounded advantage reshaping
7. PURE / Min-form PRM (NeurIPS 2025) — 解决 PRM reward hacking
8. VinePPO (ICML 2025, arxiv 2410.01679) — MC-based step credit
9. DAPO (ByteDance, NeurIPS 2025, arxiv 2503.14476) — 4 项 token-level 技术
10. Q-RM (2025) — Token Q-value as reward
11. R-PRM (EMNLP 2025) — Reasoning-driven PRM
12. GRPO (DeepSeekMath, 2024, arxiv 2402.03300) — 算法基础
13. OpenAI PRM800K (Lightman et al., 2024) — 人工标注 step reward
14. DeepSeek-R1-Zero (2025, arxiv 2501.12948) — R1-Zero 风格 RL

### MARL Reward Decomposition

15. DRBFN（你的工作）— BFN 多 agent 奖励分解
16. MAVEN — 潜空间分解
17. QPLEX — Simplex Q 分解
18. QMIX (Rashid et al., 2018) — 单调 value 分解
19. ROMA — 角色 decomposition

### 理论基础

20. Bayesian Flow Networks (Gao et al., 2023, arxiv 2308.07037) — BFN 原始论文
21. DeepSeekMath (arxiv 2402.03300) — GRPO 推导

---

**文档版本**：v1.0, 2026-07-27
**作者**：张英铭 + Claude Code 协作
**状态**：设计稿，待讨论
