# Results 数据目录

DRBFN 实验产物整理目录。**此目录在 git 中追踪**（在 `.gitignore` 中特意排除）——包含我们的论文数据。

> 重新生成所有自动产物：`python tools/extract_curves.py && python tools/plot_results.py`

---

## 目录结构

```
results/
├── README.md              # 本文件
├── tables.md              # 自动生成：每个实验的 win-rate 汇总
├── logs/                  # 原始训练日志（手工整理）
│   ├── v1/                # DRBFN v1 实验
│   ├── v2/                # DRBFN v2 实验
│   ├── v3/                # DRBFN v3 实验（当前主推）
│   └── baselines/         # MAPPO baseline 实验
├── curves/                # 自动生成：每个 log 一个 CSV（时间步 + win rate）
│   ├── <log_name>.csv
│   └── <log_name>__drbfn.csv   # DRBFN 专用统计（loss、R_mean 等）
└── figures/               # 自动生成：各地图对比图
    ├── 3m_eval_win_rate.png
    ├── 5mvs6m_eval_win_rate.png
    ├── MMM2_eval_win_rate.png
    ├── 2cvs64zg_eval_win_rate.png
    └── 2svs1sc_eval_win_rate.png
```

---

## 文件约定

### `logs/<version>/<exp_name>.log`

`python onpolicy/scripts/train/train_smac.py ...` 的原始 stdout。包含：
- 每次更新的进度：`Map X Algo Y Exp Z updates A/B episodes, total num timesteps C/D, FPS E.`
- 训练/incremental win rate：`incre win rate is X.`
- Eval win rate：`eval win rate is X.`
- DRBFN 统计：`[v3 step=S] R_mean=... drbfn_loss=...`

### `curves/<log_name>.csv`（自动）

| 列名 | 说明 |
|---|---|
| `update` | run 内的更新步数 |
| `timesteps` | 总环境时间步 |
| `fps` | 训练吞吐量 |
| `incre_win_rate_mean` | 此更新前 `incre win rate` 样本的均值 |
| `eval_win_rate` | 此更新前最后一个 `eval win rate`（可能为空） |

### `curves/<log_name>__drbfn.csv`（自动，仅 DRBFN run）

| 列名 | 说明 |
|---|---|
| `version` | 1、2 或 3 |
| `step` | 更新步数 |
| `R_mean` | 团队奖励 R 的均值 |
| `G_n_mean` | n-step 回报均值（仅 v3） |
| `n` | n-step 长度（仅 v3） |
| `q_tot_pred_mean` | 预测的 Q_tot |
| `drbfn_loss` | BFN 训练 loss |
| `qi_loss` | Per-agent Q loss |
| `qtot_loss` | 全局 Q loss |

### `figures/<map>_eval_win_rate.png`（自动）

每个 SMAC 地图一张图。每条线是一个（算法、版本、实验）序列。X 轴：时间步。Y 轴：eval win rate。

---

## 添加新实验

1. **把 log 移到正确位置**：

   ```bash
   mv /path/to/my_new_run.log results/logs/v3/
   ```

2. **刷新曲线、图表、汇总表**：

   ```bash
   python tools/extract_curves.py
   python tools/plot_results.py
   ```

3. **验证表格**：

   ```bash
   cat results/tables.md | grep my_new_run
   ```

4. **提交**：

   ```bash
   git add results/logs/v3/my_new_run.log results/curves/ results/figures/ results/tables.md
   git commit -m "results: add my_new_run on <map>"
   ```

---

## 不要提交的内容

- 原始 checkpoint 文件（`.pt`、`.pth`、`.safetensors`、`.bin`）——大文件；已被 `.gitignore` 排除
- StarCraft II 二进制文件 / 地图文件
- Conda 环境 / `__pycache__/`

如果误添加了大文件，从历史中移除：

```bash
git rm --cached path/to/large_file
git commit -m "chore: stop tracking large file"
# 可选：用 git filter-repo 清理历史
```
