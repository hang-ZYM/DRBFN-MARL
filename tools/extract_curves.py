"""
解析 SMAC 训练日志，提取学习曲线。

从 `train_smac.py` 产生的日志中提取每步数据，匹配以下格式：
  - "Map X Algo Y Exp Z updates A/B episodes, total num timesteps C/D, FPS E."
  - "incre win rate is X."          （训练/incremental win rate）
  - "eval win rate is X."           （eval win rate）
  - "[vN step=S] R_mean=... G_n_mean=... drbfn_loss=... qi_loss=... qtot_loss=..."

输出：
  results/curves/<log_name>.csv     每个 log 一个 CSV，包含所有提取的步
  results/tables.md                 所有 log 的汇总表

用法：
  python tools/extract_curves.py                    # 解析 results/logs/ 下所有 log
  python tools/extract_curves.py path/to/file.log   # 解析单个 log
  python tools/extract_curves.py --logs-dir results/logs --out-dir results
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# 正则模式
# ---------------------------------------------------------------------------

# Map 5m_vs_6m Algo r_drbfn_v3 Exp v3_5m_vs_6m_5M_seed1_full updates 4460/6250 episodes, total num timesteps 3568800/5000000, FPS 76.
PROGRESS_RE = re.compile(
    r"Map\s+(?P<map>\S+)\s+Algo\s+(?P<algo>\S+)\s+Exp\s+(?P<exp>\S+)\s+"
    r"updates\s+(?P<update>\d+)/(?P<total_updates>\d+)\s+episodes,\s+"
    r"total num timesteps\s+(?P<timesteps>\d+)/(?P<total_timesteps>\d+),\s+"
    r"FPS\s+(?P<fps>[\d.]+)"
)

# "incre win rate is 0.4012345679012346."  （注意：不能吞掉末尾的句点）
INCRE_WR_RE = re.compile(r"incre win rate is\s+(?P<win>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

# "eval win rate is 0.5."
EVAL_WR_RE = re.compile(r"eval win rate is\s+(?P<win>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

# "[v3 step=4440] R_mean=0.5417 | G_n_mean=7.2431 (n=5) | q_tot_pred_mean=7.2170 | drbfn_loss=0.0219, qi_loss=0.2376, qtot_loss=3.7975"
DRBFN_STATS_RE = re.compile(
    r"\[v(?P<version>\d+)\s+step=(?P<step>\d+)\]\s+"
    r"(?:R_mean=(?P<R_mean>[\d.eE+-]+)\s*\|\s*)?"
    r"(?:G_n_mean=(?P<G_n_mean>[\d.eE+-]+)\s*\(n=(?P<n>\d+)\)\s*\|\s*)?"
    r"(?:q_tot_pred_mean=(?P<q_tot_pred_mean>[\d.eE+-]+)\s*\|\s*)?"
    r"drbfn_loss=(?P<drbfn_loss>[\d.eE+-]+),\s*"
    r"qi_loss=(?P<qi_loss>[\d.eE+-]+),\s*"
    r"qtot_loss=(?P<qtot_loss>[\d.eE+-]+)"
)


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------

def parse_log(log_path: Path) -> dict:
    """解析单个训练日志。

    Returns:
        {
            "log_name": str,
            "map": str, "algo": str, "exp": str,
            "total_updates": int | None,
            "total_timesteps": int | None,
            "rows": list[dict],   # 每个训练更新一行
            "drbfn_rows": list[dict],   # DRBFN 专用统计（更新的子集）
        }
    """
    rows: list[dict] = []
    drbfn_rows: list[dict] = []

    map_name = algo_name = exp_name = None
    total_updates = total_timesteps = None

    # 待处理的 win rate 会被附加到下一条 progress 行
    pending_incre: list[float] = []
    pending_eval: list[float] = []

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # 匹配 progress（每个更新的标准"头"行）
            m = PROGRESS_RE.search(line)
            if m:
                map_name = m.group("map")
                algo_name = m.group("algo")
                exp_name = m.group("exp")
                total_updates = int(m.group("total_updates"))
                total_timesteps = int(m.group("total_timesteps"))

                row = {
                    "update": int(m.group("update")),
                    "timesteps": int(m.group("timesteps")),
                    "fps": float(m.group("fps")),
                    "incre_win_rate_mean": _mean_or_empty(pending_incre),
                    "eval_win_rate": pending_eval[-1] if pending_eval else "",
                }
                rows.append(row)
                pending_incre.clear()
                pending_eval.clear()
                continue

            mi = INCRE_WR_RE.search(line)
            if mi:
                pending_incre.append(float(mi.group("win")))
                continue

            me = EVAL_WR_RE.search(line)
            if me:
                pending_eval.append(float(me.group("win")))
                continue

            md = DRBFN_STATS_RE.search(line)
            if md:
                drbfn_rows.append({
                    "version": int(md.group("version")),
                    "step": int(md.group("step")),
                    "R_mean": _to_float_or_empty(md.group("R_mean")),
                    "G_n_mean": _to_float_or_empty(md.group("G_n_mean")),
                    "n": _to_int_or_empty(md.group("n")),
                    "q_tot_pred_mean": _to_float_or_empty(md.group("q_tot_pred_mean")),
                    "drbfn_loss": float(md.group("drbfn_loss")),
                    "qi_loss": float(md.group("qi_loss")),
                    "qtot_loss": float(md.group("qtot_loss")),
                })
                continue

    return {
        "log_name": log_path.stem,
        "map": map_name or "",
        "algo": algo_name or "",
        "exp": exp_name or "",
        "total_updates": total_updates,
        "total_timesteps": total_timesteps,
        "rows": rows,
        "drbfn_rows": drbfn_rows,
    }


def _mean_or_empty(xs: list[float]):
    return sum(xs) / len(xs) if xs else ""


def _to_float_or_empty(x):
    return float(x) if x is not None else ""


def _to_int_or_empty(x):
    return int(x) if x is not None else ""


# ---------------------------------------------------------------------------
# 输出写入
# ---------------------------------------------------------------------------

PROGRESS_COLS = ["update", "timesteps", "fps", "incre_win_rate_mean", "eval_win_rate"]
DRBFN_COLS = ["step", "R_mean", "G_n_mean", "n", "q_tot_pred_mean",
              "drbfn_loss", "qi_loss", "qtot_loss"]


def write_csv(parsed: dict, out_dir: Path) -> Path:
    """写入单个 log 的 progress CSV 和（如有）DRBFN 统计 CSV。

    返回 progress CSV 的路径。
    """
    stem = parsed["log_name"]
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    progress_path = curves_dir / f"{stem}.csv"
    with progress_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROGRESS_COLS)
        w.writeheader()
        for row in parsed["rows"]:
            w.writerow({k: row.get(k, "") for k in PROGRESS_COLS})

    if parsed["drbfn_rows"]:
        drbfn_path = curves_dir / f"{stem}__drbfn.csv"
        with drbfn_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["version"] + DRBFN_COLS)
            w.writeheader()
            for row in parsed["drbfn_rows"]:
                w.writerow({k: row.get(k, "") for k in ["version"] + DRBFN_COLS})

    return progress_path


def summarize(parsed: dict) -> dict:
    """计算单个解析后 log 的汇总统计。"""
    rows = parsed["rows"]
    eval_wrs = [r["eval_win_rate"] for r in rows if r["eval_win_rate"] != ""]
    incre_wrs = [r["incre_win_rate_mean"] for r in rows if r["incre_win_rate_mean"] != ""]

    last_update = rows[-1]["update"] if rows else 0
    last_timesteps = rows[-1]["timesteps"] if rows else 0
    total = parsed["total_updates"] or 0

    return {
        "log_name": parsed["log_name"],
        "map": parsed["map"],
        "algo": parsed["algo"],
        "exp": parsed["exp"],
        "last_update": last_update,
        "total_updates_target": total,
        "progress_pct": round(last_update / total * 100, 1) if total else 0,
        "last_timesteps": last_timesteps,
        "best_eval_win_rate": round(max(eval_wrs), 4) if eval_wrs else "",
        "last_eval_win_rate": round(eval_wrs[-1], 4) if eval_wrs else "",
        "n_eval_points": len(eval_wrs),
        "last_incre_win_rate": round(incre_wrs[-1], 4) if incre_wrs else "",
        "completed": last_update >= total if total else False,
    }


def write_summary_table(summaries: list[dict], out_path: Path):
    cols = [
        "log_name", "map", "algo", "exp",
        "last_update", "total_updates_target", "progress_pct",
        "best_eval_win_rate", "last_eval_win_rate",
        "n_eval_points", "last_incre_win_rate", "completed",
    ]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# 实验汇总\n\n")
        f.write("> 由 `tools/extract_curves.py` 自动生成。运行 `python tools/extract_curves.py` 刷新。\n\n")
        f.write("| Log | Map | Algo | Last Update | Progress | Best Eval WR | Last Eval WR | Eval Pts | Completed |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for s in summaries:
            completed = "yes" if s["completed"] else "partial"
            f.write(
                f"| `{s['log_name']}` | {s['map']} | {s['algo']} | "
                f"{s['last_update']}/{s['total_updates_target']} | {s['progress_pct']}% | "
                f"{s['best_eval_win_rate']} | {s['last_eval_win_rate']} | "
                f"{s['n_eval_points']} | {completed} |\n"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_logs(logs_dir: Path) -> Iterable[Path]:
    """递归遍历 logs_dir 下所有 .log 文件。"""
    yield from logs_dir.rglob("*.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", nargs="?", default=None,
                    help="单个 log 文件。若省略，则解析 --logs-dir 下所有 log。")
    ap.add_argument("--logs-dir", default="results/logs",
                    help="包含 .log 文件的目录（递归）。")
    ap.add_argument("--out-dir", default="results",
                    help="CSV 和 tables.md 的输出目录。")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs: list[Path]
    if args.log_path:
        logs = [Path(args.log_path)]
    else:
        logs = list(find_logs(Path(args.logs_dir)))
        if not logs:
            print(f"在 {args.logs_dir} 下未找到 .log 文件")
            return

    print(f"正在解析 {len(logs)} 个 log 文件...")
    summaries: list[dict] = []
    for log in logs:
        try:
            parsed = parse_log(log)
        except Exception as e:
            print(f"  [失败] {log.name}: {e}")
            continue
        if not parsed["rows"]:
            print(f"  [跳过] {log.name}: 没有 progress 行（log 不完整？）")
            continue
        csv_path = write_csv(parsed, out_dir)
        s = summarize(parsed)
        summaries.append(s)
        print(f"  [成功] {log.name}: {len(parsed['rows'])} 步 "
              f"→ {csv_path.relative_to(out_dir.parent)} | "
              f"best eval WR = {s['best_eval_win_rate']}")

    if not summaries:
        print("没有有效的 log 被解析。")
        return

    summaries.sort(key=lambda s: (s["map"], s["algo"], s["log_name"]))
    summary_path = out_dir / "tables.md"
    write_summary_table(summaries, summary_path)
    print(f"\n汇总表已写入 {summary_path}")


if __name__ == "__main__":
    main()
