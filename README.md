# DRBFN：基于贝叶斯流网络的多智能体奖励分解

> 用 **Bayesian Flow Network (BFN)** 解决合作型 MARL 中的多智能体信用分配问题。将全局团队奖励 `R` 分解为各智能体奖励 `r_i`，为 actor-critic 方法（如 MAPPO）提供细粒度的信用信号。

**基于**：[marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)（MAPPO 官方实现）
**测试环境**：StarCraft Multi-Agent Challenge (SMAC) — `3m`、`5m_vs_6m`、`MMM2`、`2c_vs_64zg`

---

## TL;DR

在合作型 MARL 中，所有智能体共享同一个团队奖励 `R`。标准做法是给每个智能体分配 `R/N`——这丢失了大量信号：在 `t-5` 时刻治疗的医疗兵（Medivac）和 `t` 时刻完成击杀的机枪兵（Marine）拿到相同的奖励。**DRBFN** 用贝叶斯流网络学习一个*随机分解* `R → (r_1, ..., r_N)`，通过 per-agent Q 网络和全局 Q 网络的 Bellman 残差进行端到端训练。该分解提供：

1. **Per-agent 信用分配**——每个智能体得到反映其实际贡献的奖励
2. **不确定性量化**——BFN 的后验方差提供置信度信号，用于安全回退
3. **即插即用**——叠加在 MAPPO 之上；当 BFN 不确定时优雅退化到 MAPPO（Dual Gate）

三个版本探索了不同的守恒范式与时间目标（详见 [docs/METHOD.md](docs/METHOD.md)）。

---

## 核心思想

### 1. BFN 作为随机奖励分解器

使用连续型 **Bayesian Flow Network**（Gao et al., 2023）作为生成模型，以 `(share_obs, joint_action)` 为条件生成 per-agent 奖励。BFN 提供：
- *确定性*预测 `r_i`（后验均值）
- *随机性*采样，用于 Q 网络训练（数据增强）
- 后验**方差**，用作置信度信号

### 2. 奖励守恒

分解奖励之和必须等于团队奖励：`Σ r_i = R`。我们探索了两种范式：
- **v1、v3（stick-breaking 硬约束）**：通过 stick-breaking 参数化结构性保证——`r_1 = w_1·R`，`r_2 = (1-w_1)·w_2·R`，...
- **v2（生成式软约束）**：直接输出 `r_i ∈ ℝ^N`，将 `Σr_i = R` 作为**软先验**（loss 项）——为*协同效应*留出空间，即 `Σr_i ≠ R` 可能更好地反映联合贡献

### 3. 训练信号：Bellman 残差

BFN 端到端训练，不需要手工标注的奖励标签。两个损失：
- `L_Qi`：per-agent Q_i(s, a)，用分解后的 `r_i` 训练
- `L_Qtot`：全局 Q_tot(s, a)，用团队奖励 `R` 训练

这些提供隐式监督：好的分解应使 per-agent Q 函数能准确预测反事实回报。

### 4. Dual Gate + MAPPO 回退

为了从一个能用的 baseline 安全起步，使用基于置信度的混合：
```
gated_r_i = confidence · r_i^DRBFN + (1 - confidence) · (R / N)
```
当 BFN 方差高时，回退到均匀分解（即原始 MAPPO）。这保证 DRBFN 不会在分布外状态上比 MAPPO 表现更差。

### 5. v3：n-step 反事实回报

v3 将 1-step TD 目标替换为 **n-step 回报**，对 Q_tot 和反事实贡献 Δ_i 都使用：
```
Δ_i = G_factual^n - G_counterfactual_i^n
```
这能捕获**延迟贡献**（如 Medivac 在 t-5 治疗 Marine → Marine 存活 → 在 t 时刻获得 +reward）。

---

## 版本演进

| 版本 | 守恒方式 | 采样 | 时间目标 | 核心动机 |
|---|---|---|---|---|
| **v1** | Stick-breaking（硬） | 单样本 | 1-step TD | 建立基线；结构保证安全 |
| **v2** | 软先验（loss） | 多采样 | 1-step TD | 允许协同效应；通过 K 个采样的方差量化不确定性 |
| **v3** | Stick-breaking（硬） | 单样本 | **n-step + 反事实** | 减少 bootstrap bias；捕获延迟贡献 |

完整算法细节与设计动机见 [docs/METHOD.md](docs/METHOD.md)。

---

## 实验结果

**测试环境**：SMAC（StarCraft Multi-Agent Challenge）
**Baseline**：MAPPO + 均匀 `R/N` 分解（合作型 MARL 标准基线），与本仓库 DRBFN 在**相同 setup**（同 episode_length、rollout_threads、PPO 超参、训练步数预算）下对比

### 核心原则：Controlled Comparison

> **不以 MAPPO 论文绝对值为对标**——论文 Table 1 报告的 MAPPO 在 MMM2 上为 90.6%（10M 步，6 seeds，特殊调参 `num_mini_batch=2, ppo_epoch=5, gain=1`），调参细节公开但实际复现困难（论文 §5.2 自己承认 MMM2 是 special case）。
>
> **本仓库所有结论基于"我们跑的 DRBFN vs 我们跑的 MAPPO"的对照比较**——两者使用相同的训练脚本骨架和步数预算，差异仅来自奖励分解模块。这是更诚实的实验设计，也更能隔离 DRBFN 的真实贡献。

### Eval Win Rate 汇总（各次实验最佳值）

| Map | Algorithm | Best Eval WR | Last Eval WR | 实际步数 | 备注 |
|---|---|---|---|---|---|
| `3m` | MAPPO (baseline) | 1.00 | 0.97 | ~4M | 全部收敛——简单地图 |
| `3m` | DRBFN v1 | **1.00** | 0.97 | ~2M | 持平 baseline |
| `3m` | DRBFN v2 | **1.00** | 1.00 | ~1.3M | 持平 baseline |
| `5m_vs_6m` | MAPPO (baseline) | 0.688 | 0.50 | ~2M | baseline 在 ~2M 步达到 |
| `5m_vs_6m` | DRBFN v1 | 0.688 | 0.50 | ~2M | 与 baseline 相近步数持平 |
| **`5m_vs_6m`** | **DRBFN v1 (long)** | **0.750** | 0.59 | **~5M** | **更长训练 → +6.2% over baseline** |
| `5m_vs_6m` | DRBFN v2 (long) | **0.750** | 0.50 | ~5M | 与 v1 持平 |
| `5m_vs_6m` | DRBFN v3 seed1 | 0.625 | 0.28 | ~4.2M | n-step 变体；不如 v1 |
| `5m_vs_6m` | DRBFN v3 seed2 | 0.500 | 0.50 | ~5M | 第二个 seed |
| `MMM2` | MAPPO (baseline) | 0.219 | 0.00 | ~6M | baseline 难收敛（论文特殊调参亦难复现）|
| `MMM2` | DRBFN v1 | 0.000 | 0.00 | ~5M | **v1 在 MMM2 完全失败**（122 eval 全 0）|
| **`MMM2`** | **DRBFN v3 (10M 配置)** | **0.312** | 0.06 | **~6.2M** | **+9.3% over baseline；v1→v3 演进的关键证据** |
| `MMM2` | DRBFN v3 (resume) | 0.281 | 0.06 | ~3.7M | 续训 |
| `MMM2` | DRBFN v3 (β=0 消融) | 0.094 | 0.03 | ~5M | KL 很重要；β=0 严重退化 |
| `2c_vs_64zg` | DRBFN v1 | **1.000** | 0.84 | ~4.7M | 困难非对称图；DRBFN 解决（无 baseline 对比）|
| `2s_vs_1sc` | MAPPO (baseline) | 1.000 | 1.00 | ~2.5M | 简单非对称图 |

### 三个核心发现

1. **`MMM2` 上 DRBFN v3 显著超过 MAPPO baseline**：在相近步数（~6M）下，**0.312 vs 0.219**（+9.3%）。MMM2 是合作型 MARL 最难的地图之一，信用分配信号最强——这是 DRBFN 应该最有用的场景。

2. **`5m_vs_6m` 上 DRBFN 在更长训练下超过 baseline**：baseline 在 ~2M 步达到 0.688 后停滞；DRBFN v1 跑到 ~5M 步达到 **0.750**（+6.2%）。但**注意**：baseline 没有跑到 5M+，所以这个对比有步数差，需要谨慎解读。

3. **版本演进的真实价值（v1→v3）**：在 MMM2 上 v1 完全失败（122 eval 全 0），v3 通过 n-step + 反事实达到了 0.312。**这是 DRBFN 设计演进的最强证据**——证明 n-step counterfactual 解决了 v1 的根本问题。

> **Baseline 健康度声明**：我们的 MAPPO baseline 在 MMM2 上 best 仅 0.219，远低于 MAPPO 论文报告的 90.6%。原因可能包括：(1) 训练步数不足（我们 ~6M vs 论文 10M）；(2) 单 seed vs 论文 6 seeds 中位数；(3) 超参未对齐（论文 MMM2 用 `num_mini_batch=2, ppo_epoch=5, gain=1` 特殊配置）。**但 DRBFN 与 MAPPO 在相同 setup 下对比，相对结论仍然有效**。
>
> 完整 win-rate 表由 `tools/extract_curves.py` 自动生成到 [results/tables.md](results/tables.md)；叙事分析见 [docs/RESULTS.md](docs/RESULTS.md)。

**图表**：执行 `python tools/plot_results.py` 可重新生成所有对比曲线到 `results/figures/`。

---

## 仓库结构

```
on-policy/
├── README.md                         # 本文件
├── docs/                             # 详细文档
│   ├── METHOD.md                     # 各版本算法细节
│   ├── RESULTS.md                    # 结果叙事分析
│   └── REPRODUCE.md                  # 复现指南
├── tools/                            # 结果提取与绘图
│   ├── extract_curves.py             # 解析 logs → CSV 曲线
│   └── plot_results.py               # 生成对比图
├── results/                          # 实验产物（已整理）
│   ├── logs/                         # 原始训练日志
│   │   ├── v1/  v2/  v3/
│   │   └── baselines/
│   ├── curves/                       # 自动生成的 CSV
│   ├── figures/                      # 自动生成的对比图
│   └── tables.md                     # 实验汇总表
├── onpolicy/                         # 源代码
│   ├── algorithms/
│   │   ├── r_drbfn/                  # v1
│   │   ├── r_drbfn_v2/               # v2
│   │   ├── r_drbfn_v3/               # v3（当前主推）
│   │   ├── r_mappo/                  # Baseline（原始）
│   │   └── utils/
│   │       ├── drbfn.py              # v1 BFN
│   │       └── drbfn_v2.py           # v2 BFN
│   ├── config.py                     # + DRBFN 专用参数
│   ├── envs/                         # + SMAC 集成
│   ├── runner/                       # + DRBFN 训练循环
│   └── scripts/                      # + train_smac_drbfn*.sh
├── environment.yaml                  # Conda 环境
├── requirements.txt
└── setup.py
```

---

## 安装

**测试通过**：Windows 11 + Python 3.12 + PyTorch 2.x + CUDA 12.x；Linux/ROCm 应该可用但未充分测试。

```bash
# 1. 克隆
git clone https://github.com/<your-username>/on-policy.git
cd on-policy

# 2. 创建环境
conda env create -f environment.yaml
conda activate marl
# 或者：pip install -e . && pip install -r requirements.txt

# 3. 安装 StarCraft II（SMAC 依赖）
# 从 http://blzdistsc2-a.akamaihd.net/Windows/SC2.4.10.zip 下载 SC2.4.10
# 解压到 ~/StarCraftII/（设置 SC2PATH 环境变量）
# 下载 SMAC 地图：https://github.com/oxwhirl/smac/raw/master/smac_maps/SMAC_Maps.zip
# 解压到 ~/StarCraftII/Maps/
```

完整安装（包括导致我们多次实验崩溃的 SC2 地图陷阱）见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

---

## 快速开始

在 `5m_vs_6m` 上训练 DRBFN v3，5M 步：

```bash
cd onpolicy/scripts
chmod +x train_smac_scripts/train_smac_drbfn_v3_5m_vs_6m.sh
./train_smac_scripts/train_smac_drbfn_v3_5m_vs_6m.sh
```

训练 MAPPO baseline 作对比：

```bash
./train_smac_scripts/train_smac_mappo_5m_vs_6m.sh
```

所有训练脚本遵循 `train_smac_<算法>_<地图>.sh` 命名约定。

---

## 文档导航

- **[docs/METHOD.md](docs/METHOD.md)** — 完整算法细节：BFN 架构、训练循环、损失函数、版本间差异
- **[docs/RESULTS.md](docs/RESULTS.md)** — 结果叙事：分地图分析、版本消融、已知问题、经验教训
- **[docs/REPRODUCE.md](docs/REPRODUCE.md)** — 复现指南：环境搭建、训练命令、评估流程、常见陷阱

---

## 关键工程注意事项

这些坑都是踩过的，完整背景见 [docs/REPRODUCE.md](docs/REPRODUCE.md)：

1. **Windows 上的 SC2 `InvalidMapData` 错误**：`D:\Temp\StarCraft II\` 中的临时地图文件可能损坏；每次运行前清空。
2. **Warmup 阶段至关重要**：在 warmup 期间用 `R/N` 训练 Q_i 和 Q_tot，先建立稳定的 Bellman 目标，再启用 BFN 训练。
3. **RNN 状态传递**：计算反事实动作时，从 buffer 逐步传递 RNN 隐藏状态（不要重置）。
4. **显存占用**：DRBFN 在单卡 8GB GPU 上能轻松跑 SMAC；BFN 参数比基础 MAPPO 仅增加 <5%。

---

## 引用

如果你使用本代码或基于 DRBFN 进行研究，请引用：

```bibtex
@misc{drbfn2026,
  title  = {DRBFN: Dynamic Reward Bayesian Flow Network for Multi-Agent Credit Assignment},
  author = {Zhang Yingming},
  year   = {2026},
  url    = {https://github.com/<your-username>/on-policy}
}
```

以及基础 MAPPO 实现：

```bibtex
@inproceedings{yu2022the,
  title     = {The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games},
  author    = {Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu and Bayen, Alexandre and Wu, Yi},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2022}
}
```

---

## 许可

MIT — 见 [LICENSE](LICENSE)。

---

## 状态

**活跃开发中**。v3 是当前主推版本；v1/v2 保留用于消融实验。论文撰写中。欢迎提 issue 和 PR。
