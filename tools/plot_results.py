"""
从已提取的 CSV 文件生成 win-rate 学习曲线图。

读取 `tools/extract_curves.py` 产生的 CSV，按地图生成对比图到 `results/figures/`。

用法：
  python tools/plot_results.py                       # 所有地图
  python tools/plot_results.py --map 5m_vs_6m        # 单个地图
  python tools/plot_results.py --curves-dir results/curves --out-dir results/figures
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def read_curve_csv(path: Path) -> list[dict]:
    """读取 extract_curves.py 产生的 progress CSV。"""
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def guess_algo_version(log_name: str) -> str:
    """从 log 名推断（算法、版本）标签。"""
    name = log_name.lower()
    if "mappo" in name:
        return "MAPPO (baseline)"
    if "v3" in name:
        return "DRBFN v3"
    if "v2" in name:
        return "DRBFN v2"
    if "v1" in name or "drbfn" in name:
        return "DRBFN v1"
    return log_name


def short_exp_name(log_name: str, map_name: str) -> str:
    """把 log 名缩短成友好的标签。"""
    s = log_name
    for prefix in [f"{map_name}_", "drbfn_", "mappo_", "v1_", "v2_", "v3_"]:
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
    return s.strip("_") or log_name


def safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_map(
    map_name: str,
    series: list[dict],
    out_path: Path,
    metric: str = "eval_win_rate",
    title_suffix: str = "",
):
    """为单个地图画一张图。

    series: list of {"label": str, "csv_path": Path}
    metric: CSV 中的列名（eval_win_rate 或 incre_win_rate_mean）
    """
    import matplotlib
    matplotlib.use("Agg")  # 无头模式
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for s in series:
        rows = read_curve_csv(s["csv_path"])
        xs, ys = [], []
        for r in rows:
            v = safe_float(r.get(metric))
            if v is None:
                continue
            xs.append(safe_float(r.get("timesteps")) or safe_float(r.get("update")) or 0)
            ys.append(v)
        if not xs:
            continue
        ax.plot(xs, ys, label=s["label"], alpha=0.85, linewidth=1.5)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Eval Win Rate" if metric == "eval_win_rate" else "Train Win Rate")
    ax.set_title(f"SMAC {map_name}{title_suffix}")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 按地图分组 CSV
# ---------------------------------------------------------------------------

MAP_RE = re.compile(r"(3m|5m_vs_6m|5m6m|MMM2|mmm2|2c_vs_64zg|2s_vs_1sc)")


def detect_map(log_name: str) -> str | None:
    """从 log 文件名推断地图名。"""
    m = MAP_RE.search(log_name)
    if not m:
        return None
    raw = m.group(1).lower()
    if raw == "5m6m":
        return "5m_vs_6m"
    if raw == "mmm2":
        return "MMM2"
    return raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves-dir", default="results/curves")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--map", default=None, help="只画这个地图（如 5m_vs_6m）")
    ap.add_argument("--metric", default="eval_win_rate",
                    choices=["eval_win_rate", "incre_win_rate_mean"])
    args = ap.parse_args()

    curves_dir = Path(args.curves_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not curves_dir.exists():
        print(f"曲线目录 {curves_dir} 不存在。请先运行 extract_curves.py。")
        return

    # 按地图分组 CSV
    by_map: dict[str, list[dict]] = {}
    for csv_path in sorted(curves_dir.glob("*.csv")):
        if csv_path.stem.endswith("__drbfn"):
            continue  # 跳过 DRBFN 逐 step loss 曲线
        map_name = detect_map(csv_path.stem)
        if not map_name:
            continue
        if args.map and map_name != args.map:
            continue
        by_map.setdefault(map_name, []).append({
            "label": f"{guess_algo_version(csv_path.stem)} · {short_exp_name(csv_path.stem, map_name)}",
            "csv_path": csv_path,
        })

    if not by_map:
        print("未找到匹配的 CSV。")
        return

    print(f"正在生成 {len(by_map)} 张图...")
    for map_name, series in by_map.items():
        # 排序：MAPPO baseline 在前，然后按版本
        series.sort(key=lambda s: (0 if "baseline" in s["label"].lower() else 1, s["label"]))
        safe_name = map_name.replace("_", "")
        out_path = out_dir / f"{safe_name}_{args.metric}.png"
        plot_map(map_name, series, out_path, metric=args.metric)
        print(f"  [成功] {map_name}: {out_path.relative_to(out_dir.parent.parent)}  （{len(series)} 条曲线）")

    print(f"\n图表已保存到 {out_dir}")


if __name__ == "__main__":
    main()
