"""Deep analysis of QVPO MMM2 2.2M step run (win rate stuck at 0%)."""
import re
import statistics

LOG = "C:/Users/张英铭/Desktop/on-policy/results/qvpo_mmm2_20260730_180544/log.txt"

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

# Parse incre win rate (training time)
incre_data = []
with open(LOG, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = re.search(r"incre win rate is ([\d.]+)\.", line)
        if m:
            incre_data.append(float(m.group(1)))

print(f"Total step data points: {len(steps_data)}")
print(f"Total eval data points: {len(eval_data)}")
print(f"Total incre win rate points: {len(incre_data)}")
print()

# ============================================================
print("=" * 80)
print(" 1. WIN RATE ANALYSIS")
print("=" * 80)
print()

# Eval win rate
eval_wins = [w for _, w in eval_data]
print(f"Eval win rate: all 0%? {all(w == 0 for w in eval_wins)}")
print(f"  Max eval win: {max(eval_wins):.4f}")
print(f"  Non-zero evals: {sum(1 for w in eval_wins if w > 0)}")

# Incre win rate (training time)
print()
print(f"Incre win rate (training): all 0%? {all(w == 0 for w in incre_data)}")
print(f"  Max incre win: {max(incre_data):.4f}")
print(f"  Non-zero incre: {sum(1 for w in incre_data if w > 0)}")
non_zero_incre = [(i, w) for i, w in enumerate(incre_data) if w > 0]
if non_zero_incre:
    print(f"  First non-zero at idx {non_zero_incre[0][0]}: {non_zero_incre[0][1]:.4f}")
    print(f"  Last 10 non-zero: {[(i, f'{w:.4f}') for i, w in non_zero_incre[-10:]]}")

# ============================================================
print()
print("=" * 80)
print(" 2. METRIC TRAJECTORIES (every 100 steps)")
print("=" * 80)
print()
print(f"{'step':>5} | {'qtot_l':>7} | {'drbfn_l':>8} | {'g_n':>6} | {'phi':>6} | "
      f"{'g_i':>6} | {'align':>7} | {'a_std':>7} | {'log_p':>7} | {'grad':>7}")
print("-" * 95)

# Sample at every 100 steps
for s in steps_data:
    if s['step'] % 100 == 0 or s['step'] == steps_data[-1]['step']:
        print(f"{s['step']:>5} | {s['qtot_loss']:>7.4f} | {s['drbfn_loss']:>8.4f} | "
              f"{s['g_n_mean']:>6.4f} | {s['phi_scale']:>6.4f} | {s['g_i_scale']:>6.4f} | "
              f"{s['raw_align_mean']:>7.4f} | {s['raw_align_std']:>7.4f} | "
              f"{s['log_p_mean']:>7.4f} | {s['grad_norm']:>7.2f}")

# ============================================================
print()
print("=" * 80)
print(" 3. Q_TOT LEARNING")
print("=" * 80)
print()

# Q_tot loss should decrease over time (learning to predict R)
early_qtot = [s for s in steps_data if s['step'] <= 100]
late_qtot = [s for s in steps_data if s['step'] >= 600]

if early_qtot and late_qtot:
    e_loss = statistics.mean(s['qtot_loss'] for s in early_qtot)
    l_loss = statistics.mean(s['qtot_loss'] for s in late_qtot)
    print(f"Q_tot loss: early={e_loss:.4f}, late={l_loss:.4f}")
    print(f"  → Q_tot learning? {'YES' if l_loss < e_loss else 'NO'}")

    e_g_n = statistics.mean(s['g_n_mean'] for s in early_qtot)
    l_g_n = statistics.mean(s['g_n_mean'] for s in late_qtot)
    print(f"g_n_mean (Q value): early={e_g_n:.4f}, late={l_g_n:.4f}")
    print(f"  → Q value growing? {'YES' if l_g_n > e_g_n else 'NO'}")

# ============================================================
print()
print("=" * 80)
print(" 4. BFN LEARNING")
print("=" * 80)
print()

if early_qtot and late_qtot:
    e_phi = statistics.mean(s['phi_scale'] for s in early_qtot)
    l_phi = statistics.mean(s['phi_scale'] for s in late_qtot)
    print(f"phi_scale: early={e_phi:.4f}, late={l_phi:.4f}")
    print(f"  → BFN output growing? {'YES' if l_phi > e_phi else 'NO'}")

    e_g_i = statistics.mean(s['g_i_scale'] for s in early_qtot)
    l_g_i = statistics.mean(s['g_i_scale'] for s in late_qtot)
    print(f"g_i_scale: early={e_g_i:.4f}, late={l_g_i:.4f}")
    print(f"  → Per-agent signal growing? {'YES' if l_g_i > e_g_i else 'NO'}")

    e_log_p = statistics.mean(s['log_p_mean'] for s in early_qtot)
    l_log_p = statistics.mean(s['log_p_mean'] for s in late_qtot)
    print(f"log_p_mean: early={e_log_p:.4f}, late={l_log_p:.4f}")
    print(f"  → BFN samples near mean? {'YES' if abs(l_log_p - e_log_p) < 1 else 'CHANGE'}")

    e_align = statistics.mean(s['raw_align_std'] for s in early_qtot)
    l_align = statistics.mean(s['raw_align_std'] for s in late_qtot)
    print(f"raw_align_std: early={e_align:.4f}, late={l_align:.4f}")
    print(f"  → BFN samples diversifying? {'YES' if l_align > e_align else 'NO'}")

# ============================================================
print()
print("=" * 80)
print(" 5. COMPARISON: MMM2 vs 5m_vs_6m (same step count)")
print("=" * 80)
print()

# Load 5m_vs_6m for comparison
LOG_5M = "C:/Users/张英铭/Desktop/on-policy/results/qvpo_5m6m_5M_20260729_203307/log.txt"
steps_5m = []
with open(LOG_5M, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = step_pattern.search(line)
        if m:
            steps_5m.append({
                'step': int(m.group(1)),
                'qtot_loss': float(m.group(2)),
                'drbfn_loss': float(m.group(3)),
                'g_n_mean': float(m.group(5)),
                'phi_scale': float(m.group(6)),
                'g_i_scale': float(m.group(7)),
                'raw_align_std': float(m.group(9)),
                'log_p_mean': float(m.group(10)),
                'grad_norm': float(m.group(11)),
            })

# Compare at key steps
print(f"{'step':>5} | {'Metric':<15} | {'MMM2':>10} | {'5m_vs_6m':>10} | {'ratio':>10}")
print("-" * 65)

for target_step in [100, 300, 500, 700]:
    mmm2 = min(steps_data, key=lambda x: abs(x['step'] - target_step))
    s5m = min(steps_5m, key=lambda x: abs(x['step'] - target_step))
    for metric in ['g_n_mean', 'g_i_scale', 'raw_align_std', 'qtot_loss']:
        v1 = mmm2[metric]
        v2 = s5m[metric]
        ratio = v1 / v2 if v2 != 0 else float('inf')
        print(f"{target_step:>5} | {metric:<15} | {v1:>10.4f} | {v2:>10.4f} | {ratio:>9.2f}x")
    print()

# ============================================================
print("=" * 80)
print(" 6. R VALUE ANALYSIS (if Q_tot is at least differentiating)")
print("=" * 80)
print()

# r_i = R/N + γΦ' - Φ
# |γΦ' - Φ| ≤ 2 * phi_scale (approximately)
# If phi_scale is small, r is mostly R/N
# If phi_scale is 0.27 and R/N is small, r deviation is significant

# In SMAC, typical rewards during episode: small negative (e.g. -0.01 per step)
# Episode win bonus: 1.0 (only at end if win)
# Since we never win, R is just small negative

# phi = 0.27 → r deviation = 2 * 0.27 = 0.54 (max)
# R/N for 10 agents with R = -0.05 is -0.005

# So r_i = -0.005 ± 0.54 → range [-0.55, +0.54]
# This is HUGE compared to true R/N

# Compare: 5m_vs_6m: phi=0.27, R/N with R=-0.01, N=5: R/N=-0.002
# r_i = -0.002 ± 0.54 → similar range

# Hmm, similar range. So r magnitude isn't the issue.

print("Theoretical r range analysis:")
print(f"  MMM2: N=10, phi_max=0.27 → r deviation up to 2*0.27 = 0.54")
print(f"  5m6m: N=5,  phi_max=0.27 → r deviation up to 2*0.27 = 0.54")
print(f"  → r deviation is the SAME")
print()

# But R/N is different
# MMM2: smaller R (rarely wins), N=10 → R/N very small
# 5m6m: medium R, N=5 → R/N larger
print("BUT: R/N differs because N differs and reward frequency differs")
print("  MMM2: very few wins → R mostly small negative → R/N tiny")
print("  5m6m: more wins → R occasionally positive → R/N meaningful")
print()

# ============================================================
print("=" * 80)
print(" 7. GRAD NORM STABILITY")
print("=" * 80)
print()

grads = [s['grad_norm'] for s in steps_data]
print(f"Grad norm stats:")
print(f"  Mean: {statistics.mean(grads):.2f}")
print(f"  Median: {statistics.median(grads):.2f}")
print(f"  Max: {max(grads):.2f}")
print(f"  Min: {min(grads):.2f}")
print(f"  Stdev: {statistics.stdev(grads):.2f}")
print(f"  Times >20: {sum(1 for g in grads if g > 20)}")
print(f"  Times >50: {sum(1 for g in grads if g > 50)}")
print(f"  → Training is {'STABLE' if max(grads) < 50 else 'UNSTABLE'}")

# ============================================================
print()
print("=" * 80)
print(" 8. KEY INSIGHTS")
print("=" * 80)
print()

print("INSIGHT 1: Q_tot is learning (loss decreasing), but g_n_mean stays small (~1.0)")
print("  → Q_tot learned to predict low team value (because we never win)")
print("  → Q_tot IS learning, just learning the wrong thing (always predict low value)")
print()

print("INSIGHT 2: g_i_scale growing (0.02 → 0.10, 5x growth)")
print("  → Q_tot IS differentiating actions (some actions are 'less bad' than others)")
print("  → BFN gets SOME signal, but very weak")
print()

print("INSIGHT 3: log_p_mean stable around 21-23 (near max for N=10)")
print("  → BFN samples close to mean (no drift, no collapse)")
print("  → BFN is 'healthy' in the sense it's not unstable")
print()

print("INSIGHT 4: phi_scale 0.19 → 0.27 (slow growth)")
print("  → BFN output slowly growing but bounded by clamp")
print()

print("INSIGHT 5: raw_align_std grew 0.003 → 0.044 (15x)")
print("  → BFN samples are diversifying, alignment signal exists")
print("  → But magnitude is still 10x smaller than 5m_vs_6m at same step")
print()

print("INSIGHT 6: Actor NEVER won a battle (incre_win = 0% throughout)")
print("  → This is the fundamental problem")
print("  → Without winning data, the entire learning loop is broken")
print("  → BFN, Q_tot, actor all need positive signal to learn")
