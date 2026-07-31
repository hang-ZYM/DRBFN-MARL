"""Analyze DRBFN-QVPO 3m 1M step results."""
import re
import os

LOG = "C:/Users/张英铭/Desktop/on-policy/results/qvpo_3m_v2_20260729_030919/log.txt"

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
                'align_mean': float(m.group(4)),
                'g_n_mean': float(m.group(5)),
                'phi_scale': float(m.group(6)),
                'g_i_scale': float(m.group(7)),
                'raw_align_mean': float(m.group(8)),
                'raw_align_std': float(m.group(9)),
                'log_p_mean': float(m.group(10)),
                'grad_norm': float(m.group(11)),
            })

# Parse eval win rate
eval_pattern = re.compile(r"eval win rate is ([\d.]+)\.")
eval_data = []
with open(LOG, 'r', encoding='utf-8', errors='ignore') as f:
    current_step = 0
    for line in f:
        # Track step from updates line
        step_match = re.search(r"updates (\d+)/", line)
        if step_match:
            current_step = int(step_match.group(1))
        m = eval_pattern.search(line)
        if m:
            eval_data.append((current_step, float(m.group(1))))

print(f"Total step data points: {len(steps_data)}")
print(f"Total eval data points: {len(eval_data)}")
print()

# ============ Win rate trajectory ============
print("=" * 80)
print(" WIN RATE TRAJECTORY")
print("=" * 80)
print(f"{'step':>6} | {'win_rate':>10}")
print("-" * 22)
for s, w in eval_data:
    print(f"{s:>6} | {w:.2%}")

# Stats
win_rates = [w for _, w in eval_data]
print()
print(f"Win rate stats:")
print(f"  Initial (warmup): {win_rates[0]:.2%}")
print(f"  Peak: {max(win_rates):.2%} @ step {eval_data[win_rates.index(max(win_rates))][0]}")
print(f"  Final: {win_rates[-1]:.2%}")
print(f"  Times reached 100%: {sum(1 for w in win_rates if w >= 0.999)}")
print(f"  Times below 90%: {sum(1 for w in win_rates if w < 0.9)}")
print(f"  Mean of last 5: {sum(win_rates[-5:])/5:.2%}")

# ============ Key metric trajectories ============
print()
print("=" * 80)
print(" BFN METRICS TRAJECTORY (every 20 steps)")
print("=" * 80)
print(f"{'step':>5} | {'phi_scale':>10} | {'g_i_scale':>10} | {'raw_align_std':>14} | {'log_p_mean':>11} | {'grad_norm':>10}")
print("-" * 75)
for d in steps_data:
    if d['step'] % 20 == 0 or d['step'] >= 300:
        print(f"{d['step']:>5} | {d['phi_scale']:>10.4f} | {d['g_i_scale']:>10.4f} | "
              f"{d['raw_align_std']:>14.4f} | {d['log_p_mean']:>11.4f} | {d['grad_norm']:>10.2f}")

# ============ Phase analysis ============
print()
print("=" * 80)
print(" PHASE ANALYSIS")
print("=" * 80)

# Phase 1: Warmup (step <= 5)
# Phase 2: Healthy learning (step 5-130)
# Phase 3: Phi explosion (step 130-160)
# Phase 4: Recovery and stability (step 170+)

phases = [
    ("Warmup", [d for d in steps_data if d['step'] <= 5]),
    ("Healthy learning", [d for d in steps_data if 5 < d['step'] <= 120]),
    ("Phi explosion", [d for d in steps_data if 120 < d['step'] <= 170]),
    ("Recovery", [d for d in steps_data if 170 < d['step'] <= 230]),
    ("Late stable", [d for d in steps_data if 230 < d['step']]),
]

for name, data in phases:
    if not data:
        continue
    print(f"\n  {name} (steps {data[0]['step']}-{data[-1]['step']}):")
    avg_phi = sum(d['phi_scale'] for d in data) / len(data)
    avg_g_i = sum(d['g_i_scale'] for d in data) / len(data)
    avg_log_p = sum(d['log_p_mean'] for d in data) / len(data)
    avg_grad = sum(d['grad_norm'] for d in data) / len(data)
    print(f"    avg phi_scale:    {avg_phi:.4f}")
    print(f"    avg g_i_scale:    {avg_g_i:.4f}")
    print(f"    avg log_p_mean:   {avg_log_p:.4f}")
    print(f"    avg grad_norm:    {avg_grad:.2f}")

# ============ Q_tot learning ============
print()
print("=" * 80)
print(" Q_TOT LEARNING")
print("=" * 80)
qtot_early = [d for d in steps_data if d['step'] <= 30]
qtot_late = [d for d in steps_data if d['step'] >= 280]
if qtot_early and qtot_late:
    print(f"  Early qtot_loss avg:  {sum(d['qtot_loss'] for d in qtot_early)/len(qtot_early):.4f}")
    print(f"  Late qtot_loss avg:   {sum(d['qtot_loss'] for d in qtot_late)/len(qtot_late):.4f}")
    print(f"  Early g_n_mean avg:   {sum(d['g_n_mean'] for d in qtot_early)/len(qtot_early):.4f}")
    print(f"  Late g_n_mean avg:    {sum(d['g_n_mean'] for d in qtot_late)/len(qtot_late):.4f}")
    print(f"  Q_tot grew by: {sum(d['g_n_mean'] for d in qtot_late)/len(qtot_late) / (sum(d['g_n_mean'] for d in qtot_early)/len(qtot_early)):.2f}x")

# ============ BFN credit assignment effectiveness ============
print()
print("=" * 80)
print(" BFN CREDIT ASSIGNMENT EFFECTIVENESS")
print("=" * 80)
# Compare phi_scale vs g_i_scale — they should grow together if BFN tracks g
print(f"  phi_scale growth:  {steps_data[0]['phi_scale']:.4f} → {steps_data[-1]['phi_scale']:.4f} ({steps_data[-1]['phi_scale']/max(steps_data[0]['phi_scale'], 1e-6):.1f}x)")
print(f"  g_i_scale growth:  {steps_data[0]['g_i_scale']:.4f} → {steps_data[-1]['g_i_scale']:.4f} ({steps_data[-1]['g_i_scale']/max(steps_data[0]['g_i_scale'], 1e-6):.1f}x)")
print(f"  raw_align_std:     {steps_data[0]['raw_align_std']:.4f} → {steps_data[-1]['raw_align_std']:.4f} ({steps_data[-1]['raw_align_std']/max(steps_data[0]['raw_align_std'], 1e-6):.1f}x)")

# Ratio: phi_scale / g_i_scale (should be ~constant if BFN scales Φ with g)
phi_over_g_early = sum(d['phi_scale']/max(d['g_i_scale'], 1e-6) for d in qtot_early)/len(qtot_early)
phi_over_g_late = sum(d['phi_scale']/max(d['g_i_scale'], 1e-6) for d in qtot_late)/len(qtot_late)
print(f"\n  phi/g_i ratio early: {phi_over_g_early:.2f}")
print(f"  phi/g_i ratio late:  {phi_over_g_late:.2f}")
print(f"  (if BFN tracks g well, ratio should be similar)")
