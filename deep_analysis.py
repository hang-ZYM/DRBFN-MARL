"""Deep analysis of QVPO 5m_vs_6m 5M run."""
import re
import statistics

LOG = "C:/Users/张英铭/Desktop/on-policy/results/qvpo_5m6m_5M_20260729_203307/log.txt"

# Parse all step data
steps_data = []
step_pattern = re.compile(
    r"\[v_qvpo step=(\d+)\] qtot_loss=([-\d.]+) \| drbfn_loss=([-\d.]+) \| "
    r"align_mean=([-\d.]+) \| g_n_mean=([-\d.]+) \| phi_scale=([-\d.]+) \| "
    r"g_i_scale=([-\d.]+) \| raw_align_mean=([-\d.]+) \| raw_align_std=([-\d.]+) \| "
    r"log_p_mean=([-\d.]+) \| grad_norm=([-\d.]+)"
)
with open(LOG, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = step_pattern.search(line)
        if m:
            steps_data.append({
                'step': int(m.group(1)),
                'qtot_loss': float(m.group(2)),
                'drbfn_loss': float(m.group(3)),
                'g_n_mean': float(m.group(5)),
                'phi_scale': float(m.group(6)),
                'g_i_scale': float(m.group(7)),
                'raw_align_mean': float(m.group(8)),
                'raw_align_std': float(m.group(9)),
                'log_p_mean': float(m.group(10)),
                'grad_norm': float(m.group(11)),
            })

# Parse eval data
eval_data = []
current_update = 0
eval_pattern = re.compile(r"eval win rate is ([\d.]+)\.")
update_pattern = re.compile(r"updates (\d+)/")
with open(LOG, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = update_pattern.search(line)
        if m:
            current_update = int(m.group(1))
        m = eval_pattern.search(line)
        if m:
            eval_data.append((current_update, float(m.group(1))))

print(f"Total step data points: {len(steps_data)}")
print(f"Total eval data points: {len(eval_data)}")
print()

# ============================================================
print("=" * 80)
print(" PHASE 1: WIN RATE DETAILED ANALYSIS")
print("=" * 80)
print()

# Define phases based on win rate trajectory
# Look at the data and identify natural phases
win_rates = [w for _, w in eval_data]

# Find peak
peak_idx = win_rates.index(max(win_rates))
peak_update, peak_win = eval_data[peak_idx]
print(f"PEAK: {peak_win:.2%} @ update {peak_update}")
print()

# Find first time >=50%, >=70%, >=80%, >=90%
first_50 = next((u for u, w in eval_data if w >= 0.5), None)
first_70 = next((u for u, w in eval_data if w >= 0.7), None)
first_80 = next((u for u, w in eval_data if w >= 0.8), None)
first_90 = next((u for u, w in eval_data if w >= 0.9), None)

print(f"First >=50%: update {first_50}")
print(f"First >=70%: update {first_70}")
print(f"First >=80%: update {first_80}")
print(f"First >=90%: update {first_90}")
print()

# Phase definition (based on actual data)
# Phase 1: warmup (0-25)
# Phase 2: rapid learning (50-300) — first to 80%
# Phase 3: peak (300-700) — including 90% peak
# Phase 4: stability (700-1000) — 70-80% range
# Phase 5: degradation (1000-1450) — drop to 60% range

phases = [
    ("Phase 1: Warmup", 0, 25),
    ("Phase 2: Rapid learning", 25, 300),
    ("Phase 3: Peak (90%+)", 300, 700),
    ("Phase 4: Stable high", 700, 1000),
    ("Phase 5: Late degradation", 1000, 1500),
]

print("PHASE BREAKDOWN:")
print(f"{'Phase':<30} | {'#evals':>6} | {'mean':>6} | {'min':>6} | {'max':>6}")
print("-" * 70)
for name, lo, hi in phases:
    phase_evals = [(u, w) for u, w in eval_data if lo <= u < hi]
    if phase_evals:
        ws = [w for _, w in phase_evals]
        print(f"{name:<30} | {len(ws):>6} | {statistics.mean(ws):>6.2%} | "
              f"{min(ws):>6.2%} | {max(ws):>6.2%}")

# ============================================================
print()
print("=" * 80)
print(" PHASE 2: BFN LEARNING DYNAMICS")
print("=" * 80)
print()

# BFN metrics by phase
print(f"{'Phase':<30} | {'phi':>6} | {'g_i':>6} | {'log_p':>7} | {'grad':>7} | {'align_std':>10}")
print("-" * 85)
for name, lo, hi in phases:
    phase_steps = [s for s in steps_data if lo <= s['step'] < hi * (1562/1455)]
    if not phase_steps:
        # try direct step range
        phase_steps = [s for s in steps_data if lo <= s['step'] <= hi]
    if phase_steps:
        avg_phi = statistics.mean(s['phi_scale'] for s in phase_steps)
        avg_g_i = statistics.mean(s['g_i_scale'] for s in phase_steps)
        avg_log_p = statistics.mean(s['log_p_mean'] for s in phase_steps)
        avg_grad = statistics.mean(s['grad_norm'] for s in phase_steps)
        avg_align = statistics.mean(s['raw_align_std'] for s in phase_steps)
        print(f"{name:<30} | {avg_phi:>6.3f} | {avg_g_i:>6.3f} | "
              f"{avg_log_p:>7.2f} | {avg_grad:>7.2f} | {avg_align:>10.3f}")

# ============================================================
print()
print("=" * 80)
print(" PHASE 3: CRITICAL TRANSITIONS")
print("=" * 80)
print()

# Find when log_p started dropping significantly
print("Log_p trajectory (key transitions):")
print(f"{'step':>5} | {'log_p':>7} | {'grad':>7} | {'win':>6}")
print("-" * 35)
# Sample at regular intervals
for s in steps_data:
    if s['step'] in [50, 100, 200, 300, 500, 700, 900, 1000, 1100, 1200, 1300, 1400]:
        # Find closest eval
        closest_eval = min(eval_data, key=lambda x: abs(x[0] - s['step']))
        win_str = f"{closest_eval[1]:.2%}" if abs(closest_eval[0] - s['step']) < 50 else "  -"
        print(f"{s['step']:>5} | {s['log_p_mean']:>7.2f} | {s['grad_norm']:>7.2f} | {win_str:>6}")

# ============================================================
print()
print("=" * 80)
print(" PHASE 4: BFN CREDIT ASSIGNMENT EFFECTIVENESS")
print("=" * 80)
print()

# Key growth metrics
print("Growth trajectory (start → peak → end):")
print(f"{'Metric':<20} | {'step 50':>8} | {'step 625 (peak)':>15} | {'step 1450 (end)':>15} | {'growth':>10}")
print("-" * 80)

def get_metric_at(step_target):
    closest = min(steps_data, key=lambda x: abs(x['step'] - step_target))
    return closest

s_start = get_metric_at(50)
s_peak = get_metric_at(625)
s_end = get_metric_at(1450)

metrics = ['phi_scale', 'g_i_scale', 'raw_align_std', 'g_n_mean', 'qtot_loss', 'log_p_mean']
for m in metrics:
    v1 = s_start[m]
    v2 = s_peak[m]
    v3 = s_end[m]
    growth = v3 / v1 if v1 != 0 else float('inf')
    print(f"{m:<20} | {v1:>8.4f} | {v2:>15.4f} | {v3:>15.4f} | {growth:>9.2f}x")

# ============================================================
print()
print("=" * 80)
print(" PHASE 5: STABILITY EVENTS")
print("=" * 80)
print()

# Find all "low" evals (<50%)
print("Eval win rates below 50%:")
print(f"{'#':>3} | {'update':>6} | {'win':>6}")
print("-" * 22)
for i, (u, w) in enumerate(eval_data):
    if w < 0.5:
        print(f"{i+1:>3} | {u:>6} | {w:.2%}")

print()
print("Eval win rates >= 80%:")
print(f"{'#':>3} | {'update':>6} | {'win':>6}")
print("-" * 22)
for i, (u, w) in enumerate(eval_data):
    if w >= 0.8:
        print(f"{i+1:>3} | {u:>6} | {w:.2%}")

# Find grad_norm spikes
print()
print("Grad norm spikes (>20):")
print(f"{'step':>5} | {'grad_norm':>10} | {'log_p':>7} | {'phi':>6}")
print("-" * 38)
spike_count = 0
for s in steps_data:
    if s['grad_norm'] > 20:
        print(f"{s['step']:>5} | {s['grad_norm']:>10.2f} | {s['log_p_mean']:>7.2f} | {s['phi_scale']:>6.3f}")
        spike_count += 1
print(f"\nTotal spikes (>20): {spike_count}")

# ============================================================
print()
print("=" * 80)
print(" PHASE 6: COMPARISON WITH BASELINES")
print("=" * 80)
print()

# Compute additional metrics
mean_last_5 = sum(win_rates[-5:]) / 5
mean_last_10 = sum(win_rates[-10:]) / 10
mean_last_20 = sum(win_rates[-20:]) / 20

# For comparison, also compute "peak phase" mean (around peak)
peak_phase = [w for u, w in eval_data if 500 <= u <= 750]
peak_phase_mean = sum(peak_phase) / len(peak_phase) if peak_phase else 0

# Stable phase mean (700-1000)
stable_phase = [w for u, w in eval_data if 700 <= u <= 1000]
stable_phase_mean = sum(stable_phase) / len(stable_phase) if stable_phase else 0

print(f"{'Metric':<25} | {'QVPO':>10} | {'v1':>10} | {'MAPPO':>10}")
print("-" * 65)
print(f"{'Peak':<25} | {max(win_rates):>10.2%} | {'75%':>10} | {'68.75%':>10}")
print(f"{'Mean last 5':<25} | {mean_last_5:>10.2%} | {'-':>10} | {'-':>10}")
print(f"{'Mean last 10':<25} | {mean_last_10:>10.2%} | {'60.9%':>10} | {'47.5%':>10}")
print(f"{'Mean last 20':<25} | {mean_last_20:>10.2%} | {'-':>10} | {'-':>10}")
print(f"{'Mean peak phase (500-750)':<25} | {peak_phase_mean:>10.2%} | {'-':>10} | {'-':>10}")
print(f"{'Mean stable phase (700-1000)':<25} | {stable_phase_mean:>10.2%} | {'-':>10} | {'-':>10}")
print(f"{'Times >=80%':<25} | {sum(1 for w in win_rates if w >= 0.8):>10} | {'-':>10} | {'-':>10}")
print(f"{'Times >=70%':<25} | {sum(1 for w in win_rates if w >= 0.7):>10} | {'-':>10} | {'-':>10}")
