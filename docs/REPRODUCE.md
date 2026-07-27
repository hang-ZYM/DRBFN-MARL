# 复现指南

在 SMAC 上复现 DRBFN 结果的分步说明。

---

## 1. 环境搭建

### 1.1 硬件

- **GPU**：1 张 ≥ 8 GB 显存（在 RTX 5060 8GB、RTX 3090、A100 上测试通过）
- **内存**：≥ 16 GB
- **磁盘**：≥ 50 GB 可用空间（StarCraft II + checkpoint）

### 1.2 操作系统

- **推荐**：Ubuntu 22.04+（SC2 稳定性最佳）
- **可用**：Windows 11（有限制——见 [§4 常见陷阱](#4-常见陷阱)）
- **未测试**：macOS

### 1.3 Conda 环境

```bash
git clone https://github.com/<your-username>/on-policy.git
cd on-policy
conda env create -f environment.yaml
conda activate marl
pip install -e .
```

如果 `environment.yaml` 过时，核心包是：

```bash
pip install torch numpy scipy seaborn pytest pysc2 s2clientprotocol gym==0.21.0
```

### 1.4 StarCraft II 安装

```bash
# Linux
wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip SC2.4.10.zip -d ~/
echo 'export SC2PATH=~/StarCraftII' >> ~/.bashrc
source ~/.bashrc

# SMAC 地图（必需）
git clone https://github.com/oxwhirl/smac.git
cp -r smac/smac_maps/SMAC_Maps ~/StarCraftII/Maps/

# Stable ID（推荐）
cp smac/stableid.json ~/StarCraftII/
```

Windows 上：下载 [SC2.4.10 Windows zip](http://blzdistsc2-a.akamaihd.net/Windows/SC2.4.10.zip)，解压到如 `C:\Program Files\StarCraft II\`，然后设置 `SC2PATH` 环境变量。

验证：

```bash
python -c "from pysc2 import maps; print('SMAC OK' if maps.get('3m') else 'MISSING')"
```

---

## 2. 训练

### 2.1 DRBFN v1 在 `5m_vs_6m`（10M 步，核心结果）

```bash
cd onpolicy/scripts
chmod +x train_smac_scripts/train_smac_drbfn_5m_vs_6m_10M.sh
./train_smac_scripts/train_smac_drbfn_5m_vs_6m_10M.sh
```

预计耗时：单卡 A100 约 12-18 小时，RTX 5060 约 30+ 小时。

### 2.2 DRBFN v3 在 `MMM2`（10M 步，核心结果）

```bash
chmod +x train_smac_scripts/train_smac_drbfn_v3_MMM2_10M.sh
./train_smac_scripts/train_smac_drbfn_v3_MMM2_10M.sh
```

预计耗时：A100 约 15-20 小时（v3 的反事实 rollout 增加约 30% 开销）。

### 2.3 MAPPO baseline

```bash
chmod +x train_smac_scripts/train_smac_mappo_<map>.sh
./train_smac_scripts/train_smac_mappo_<map>.sh
```

### 2.4 自定义配置

所有 DRBFN 超参数都通过 CLI 参数暴露（见 `onpolicy/config.py`）。关键的：

```bash
python train_smac.py \
    --algorithm r_drbfn_v3 \
    --experiment_name my_drbfn_v3_run \
    --map_name MMM2 \
    --num_env_steps 10000000 \
    --drbfn_warmup_t 10000 \
    --drbfn_n_step 5 \
    --drbfn_beta 0.04 \
    --drbfn_cf_temperature 1.0
```

---

## 3. 评估

评估在训练中自动进行（默认：每 5000 时间步，32 个 eval episode）。要独立运行评估：

```bash
python onpolicy/scripts/eval/eval_smac.py \
    --model_dir results/StarCraft2/MMM2/r_drbfn_v3/run1 \
    --eval_episodes 100
```

---

## 4. 常见陷阱

### 4.1 SC2 `InvalidMapData` 错误（Windows 特有）

**症状**：

```
pysc2.lib.remote_controller.RequestError: SC2APIProtocol.ResponseCreateGame.Error.InvalidMapData:
'temporary map 'D:\Temp\StarCraft II\TempLaunchMap.SC2Map' has invalid data.'
```

**原因**：SC2 每局游戏会存储一份临时地图副本。在 Windows 上，这些文件在多次运行间可能损坏（清理时的竞争条件）。

**修复**：

```bash
# 每次训练前
rm -rf "/d/Temp/StarCraft II/"
mkdir -p "/d/Temp/StarCraft II/"
```

或者把 `SC2_TEMP_MAP_DIR` 设到一个全新目录：

```bash
export SC2_TEMP_MAP_DIR=/tmp/sc2_maps_$$
```

**建议**：尽可能用 Linux。我们因此问题在 Windows 上损失了多次 10M 步的 run。

### 4.2 DRBFN 发散（BFN 产生 NaN 奖励）

**症状**：`drbfn_loss` 是 NaN；奖励变 NaN；训练崩溃。

**原因**：BFN 在 warmup 结束前就被训练，或学习率太高。

**修复**：

```bash
# 增加 warmup
--drbfn_warmup_t 50000

# 降低 BFN 专用学习率（如果暴露）
--drbfn_lr 1e-4
```

### 4.3 DRBFN 在短训练（< 5M）时不如 MAPPO

**预期行为**。DRBFN 的优势只在 5M 步之后显现。见 [docs/RESULTS.md §版本消融](RESULTS.md#版本消融)。

### 4.4 v3 反事实 rollout 期间 GPU OOM

**症状**：`_compute_gated_rewards` 期间 CUDA OOM。

**原因**：v3 的反事实 rollout 每个训练步需要 N 次前向传播（每个智能体一次）。

**修复**：

```bash
# 减少 rollout 长度
--drbfn_n_step 3  # 默认 5

# 或者批量处理反事实（代码修改；见 r_drbfn_v3.py:_compute_gated_rewards）
```

### 4.5 `vllm` 导入错误

与 DRBFN 无关（仅用于 LLM RL 工作）。如果看到此错误，说明误装了 vLLM 相关依赖。删除它们：

```bash
pip uninstall vllm
```

---

## 5. 重新提取结果

跑完新实验后，刷新结果：

```bash
# 1. 把新 log 移到 results/logs/<version>/
mv my_new_run.log results/logs/v3/

# 2. 重新提取曲线
python tools/extract_curves.py

# 3. 重新生成图表
python tools/plot_results.py

# 4. 提交
git add results/
git commit -m "results: add my_new_run on <map>"
```

`results/tables.md` 和 `results/figures/*.png` 是自动生成的——不要手工编辑。要改图表样式，编辑 `tools/plot_results.py`。

---

## 6. 验证安装

最小化自检：

```bash
python -c "
import torch
from onpolicy.algorithms.r_drbfn.r_drbfn import R_DRBFN
from onpolicy.algorithms.utils.drbfn import UnifiedWorldModel
print('DRBFN v1 imports OK')
from onpolicy.algorithms.r_drbfn_v3.r_drbfn_v3 import R_DRBFN_v3
print('DRBFN v3 imports OK')
"
```

1 分钟训练烟雾测试（3m 上跑 1000 步）：

```bash
python onpolicy/scripts/train/train_smac.py \
    --algorithm r_drbfn \
    --experiment_name smoke_test \
    --map_name 3m \
    --num_env_steps 1000
```

如果无错误完成并产生 `results/StarCraft2/3m/r_drbfn/smoke_test/`，说明你的环境搭建正确。

---

## 7. 运行后的文件结构

```
results/
├── logs/
│   ├── v1/
│   ├── v2/
│   ├── v3/
│   └── baselines/
├── curves/                          # 自动：每个 log 一个 CSV
│   ├── drbfn_5m6m_10M.csv
│   ├── mappo_5m6m_baseline.csv
│   └── ...
├── figures/                         # 自动：各地图 win-rate 曲线
│   ├── 3m_eval_win_rate.png
│   ├── 5mvs6m_eval_win_rate.png
│   ├── MMM2_eval_win_rate.png
│   └── ...
└── tables.md                        # 自动：实验汇总

onpolicy/scripts/results/            # 默认训练输出目录
└── StarCraft2/
    └── <map>/<algo>/<exp>/
        ├── checkpoints/
        ├── run1/logs/...
        └── config.json
```

---

## 8. 获取帮助

- **算法问题**：读 [docs/METHOD.md](METHOD.md)
- **结果解读**：读 [docs/RESULTS.md](RESULTS.md)
- **Bug**：在 GitHub 上提 issue，附上 `results/logs/` 中相关 log 文件
