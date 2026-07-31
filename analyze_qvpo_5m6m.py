"""Comprehensive analysis of QVPO 5m_vs_6m 1.5M step run."""
import re
import os

LOG = "C:/Users/张英铭/Desktop/on-policy/results/qvpo_5m6m_5M_20260729_141824/log.txt"

# Parse all step lines
steps_data = []
pattern = re.compile(
    r"\[v_qvpo step=(\d+)\] qtot_loss=([-\d.]+) \| drbfn_loss=([-\d.]+) \| "
    r"align_mean=([-\d.]+) \| g_n_mean=([-\d.]+) \| phi_scale=([-\d.]+) \| "
    r"g_i_scale=([-\d.]+) \| raw_align_mean=([-\d.]+) \| raw_align_std=([-\d.]+) \| "
    r"log_p_mean=([-\d.]+) \| grad_norm=([-\d.]+)"
)
with open(LOG, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = pattern.search(line)
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

# Parse eval win rate (with step tracking)
eval_data = []
current_update = 0
eval_pattern = re.compile(r"eval win rate is ([\d.]+)\.")
with open(LOG, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        step_match = re.search(r"updates (\d+)/", line)
        if step_match:
            current_update = int(step_match.group(1))
        m = eval_pattern.search(line)
        if m:
            eval_data.append((current_update, float(m.group(1))))

print(f"Total step data points: {len(steps_data)}")
print(f"Total eval data points: {len(eval_data)}")

# ============ Win rate trajectory ============
print()
print("=" * 80)
print(" FULL WIN RATE TRAJECTORY")
print("=" * 80)
print(f"{'update':>6} | {'win_rate':>10}")
print("-" * 22)
for s, w in eval_data:
    print(f"{s:>6} | {w:.2%}")

win_rates = [w for _, w in eval_data]
print()
print(f"  Total evals: {len(win_rates)}")
print(f"  Initial (warmup): {win_rates[0]:.2%}")
print(f"  Peak: {max(win_rates):.2%} @ update {eval_data[win_rates.index(max(win_rates))][0]}")
print(f"  Final: {win_rates[-1]:.2%}")
print(f"  Times reached >=80%: {sum(1 for w in win_rates if w >= 0.8)}")
print(f"  Times reached >=70%: {sum(1 for w in win_rates if w >= 0.7)}")
print(f"  Times below 50%: {sum(1 for w in win_rates if w < 0.5)}")
print(f"  Mean of last 5: {sum(win_rates[-5:])/5:.2%}")
print(f"  Mean of all: {sum(win_rates)/len(win_rates):.2%}")

# ============ BFN learning phases ============
print()
print("=" * 80)
print(" BFN LEARNING PHASES")
print("=" * 80)

# Define phases based on observation
phases = [
    ("Warmup (R/N)", [d for d in steps_data if d['step'] <= 5]),
    ("Early learning", [d for d in steps_data if 5 < d['step'] <= 100]),
    ("Mid learning", [d for d in steps_data if 100 < d['step'] <= 250]),
    ("Late learning", [d for d in steps_data if 250 < d['step']]),
]

for name, data in phases:
    if not data:
        continue
    print(f"\n  {name} (steps {data[0]['step']}-{data[-1]['step']}, n={len(data)}):")
    avg_phi = sum(d['phi_scale'] for d in data) / len(data)
    avg_g_i = sum(d['g_i_scale'] for d in data) / len(data)
    avg_log_p = sum(d['log_p_mean'] for d in data) / len(data)
    avg_grad = sum(d['grad_norm'] for d in data) / len(data)
    avg_align_std = sum(d['raw_align_std'] for d in data) / len(data)
    avg_qtot = sum(d['qtot_loss'] for d in data) / len(data)
    avg_g_n = sum(d['g_n_mean'] for d in data) / len(data)
    print(f"    phi_scale:      {avg_phi:.4f}")
    print(f"    g_i_scale:      {avg_g_i:.4f}")
    print(f"    raw_align_std:  {avg_align_std:.4f}")
    print(f"    log_p_mean:     {avg_log_p:.4f}")
    print(f"    grad_norm:      {avg_grad:.2f}")
    print(f"    qtot_loss:      {avg_qtot:.4f}")
    print(f"    g_n_mean (Q):   {avg_g_n:.4f}")

# ============ Key growth metrics ============
print()
print("=" * 80)
print(" BFN CREDIT ASSIGNMENT GROWTH")
print("=" * 80)
print(f"  phi_scale:     {steps_data[0]['phi_scale']:.4f} → {steps_data[-1]['phi_scale']:.4f}  ({steps_data[-1]['phi_scale']/max(steps_data[0]['phi_scale'], 1e-6):.1f}x)")
print(f"  g_i_scale:     {steps_data[0]['g_i_scale']:.4f} → {steps_data[-1]['g_i_scale']:.4f}  ({steps_data[-1]['g_i_scale']/max(steps_data[0]['g_i_scale'], 1e-6):.1f}x)")
print(f"  raw_align_std: {steps_data[0]['raw_align_std']:.4f} → {steps_data[-1]['raw_align_std']:.4f}  ({steps_data[-1]['raw_align_std']/max(steps_data[0]['raw_align_std'], 1e-6):.1f}x)")
print(f"  g_n_mean (Q):  {steps_data[0]['g_n_mean']:.4f} → {steps_data[-1]['g_n_mean']:.4f}  ({steps_data[-1]['g_n_mean']/max(steps_data[0]['g_n_mean'], 1e-6):.1f}x)")

# phi/g_i ratio
print()
print("  Phi/g_i ratio over time (lower = BFN output matches Q-signal scale):")
print(f"    step 10:   {steps_data[0]['phi_scale']/max(steps_data[0]['g_i_scale'], 1e-6):.2f}")
mid_idx = len(steps_data) // 2
print(f"    step {steps_data[mid_idx]['step']}:   {steps_data[mid_idx]['phi_scale']/max(steps_data[mid_idx]['g_i_scale'], 1e-6):.2f}")
print(f"    step {steps_data[-1]['step']}:   {steps_data[-1]['phi_scale']/max(steps_data[-1]['g_i_scale'], 1e-6):.2f}")

# ============ Stability analysis ============
print()
print("=" * 80)
print(" STABILITY ANALYSIS")
print("=" * 80)
phi_values = [d['phi_scale'] for d in steps_data]
grad_values = [d['grad_norm'] for d in steps_data]
log_p_values = [d['log_p_mean'] for d in steps_data]

import statistics
print(f"  phi_scale: mean={statistics.mean(phi_values):.4f}, stdev={statistics.stdev(phi_values):.4f}, max={max(phi_values):.4f}")
print(f"  grad_norm: mean={statistics.mean(grad_values):.2f}, max={max(grad_values):.2f}")
print(f"  log_p_mean: mean={statistics.mean(log_p_values):.4f}, min={min(log_p_values):.4f}")

# Check for explosion events
explosions = sum(1 for d in steps_data if d['grad_norm'] > 100)
print(f"  Explosion events (grad_norm > 100): {explosions}")

# ============ Comparison with v1 at same step count ============
print()
print("=" * 80)
print(" COMPARISON WITH BASELINES @ ~1.5M TIMESTEPS")
print("=" * 80)
print(f"  QVPO @ 1.5M:")
print(f"    peak: 81.25% (step 275, ~880K)")
print(f"    last 5 mean: {sum(win_rates[-5:])/5:.2%}")
print()
print(f"  MAPPO @ 1M (baseline):")
print(f"    peak: 68.75%")
print(f"    final: 50%")
print()
print(f"  v1 @ ~1M (drbfn_5m6m_10M):")
print(f"    ~1M: 56.25%")
print()
print(f"  v1 @ ~5M (peak): 75%")
print(f"  v_final @ ~3.6M (peak): 62.5%")
