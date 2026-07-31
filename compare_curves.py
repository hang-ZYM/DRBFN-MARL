"""Extract MAPPO baseline curves and compare with QVPO."""
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_curve(log_dir, tag='eval_win_rate'):
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            return [(e.step, e.value) for e in events]
    except Exception as e:
        return []
    return []

def print_curve_at_milestones(curve, name, milestones=None):
    if milestones is None:
        milestones = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
    print(f'\n=== {name} ===')
    if not curve:
        print('  no data')
        return
    print(f'  total points: {len(curve)}, final step: {curve[-1][0]:,}')
    print(f'  {"step":>10} | eval_winrate')
    print(f'  {"-"*10}-+-{"-"*15}')

    # Print at milestone steps (find closest)
    for ms in milestones:
        closest = min(curve, key=lambda c: abs(c[0] - ms))
        if abs(closest[0] - ms) < 50_000:  # within 50k
            print(f'  {closest[0]:>10,} | {closest[1]:.2%}')

    # Also print every 10th point if curve is short
    if len(curve) < 50:
        print('  ---all points---')
        for s, v in curve:
            print(f'  {s:>10,} | {v:.2%}')


# 3m experiments
print('=' * 70)
print(' 3m MAPPO baselines')
print('=' * 70)

mappo_3m_dirs = [
    ('MAPPO baseline (long)', 'C:/Users/张英铭/Desktop/on-policy/onpolicy/scripts/results/StarCraft2/3m/rmappo/mappo_3m/run1/logs/eval_win_rate/eval_win_rate'),
    ('MAPPO baseline (500k)', 'C:/Users/张英铭/Desktop/on-policy/onpolicy/scripts/results/StarCraft2/3m/rmappo/mappo_3m_baseline/run1/logs/eval_win_rate/eval_win_rate'),
]

for name, log_dir in mappo_3m_dirs:
    if os.path.exists(log_dir):
        curve = get_curve(log_dir)
        print_curve_at_milestones(curve, name)
    else:
        print(f'\n=== {name} ===\n  NOT FOUND: {log_dir}')

# QVPO 3m for comparison
qvpo_3m_dir = 'C:/Users/张英铭/Desktop/on-policy/onpolicy/scripts/results/StarCraft2/3m/r_drbfn_qvpo/qvpo_3m/run1/logs/eval_win_rate/eval_win_rate'
if os.path.exists(qvpo_3m_dir):
    curve = get_curve(qvpo_3m_dir)
    print_curve_at_milestones(curve, 'QVPO 3m (500k)')

# v1 3m for comparison
v1_3m_dir = 'C:/Users/张英铭/Desktop/on-policy/onpolicy/scripts/results/StarCraft2/3m/r_drbfn/drbfn_3m_full/run1/logs/eval_win_rate/eval_win_rate'
if os.path.exists(v1_3m_dir):
    curve = get_curve(v1_3m_dir)
    print_curve_at_milestones(curve, 'v1 3m (2M)')

# 5m_vs_6m
print('\n')
print('=' * 70)
print(' 5m_vs_6m baselines')
print('=' * 70)

mappo_5m6m_dirs = [
    ('MAPPO 5m_vs_6m baseline', 'C:/Users/张英铭/Desktop/on-policy/onpolicy/scripts/results/StarCraft2/5m_vs_6m/rmappo/mappo_5m6m_baseline/run2/logs/eval_win_rate/eval_win_rate'),
]
for name, log_dir in mappo_5m6m_dirs:
    if os.path.exists(log_dir):
        curve = get_curve(log_dir)
        print_curve_at_milestones(curve, name)

# v1 5m6m
v1_5m6m_dir = 'C:/Users/张英铭/Desktop/on-policy/onpolicy/scripts/results/StarCraft2/5m_vs_6m/r_drbfn/drbfn_5m6m_10M/run1/logs/eval_win_rate/eval_win_rate'
if os.path.exists(v1_5m6m_dir):
    curve = get_curve(v1_5m6m_dir)
    print_curve_at_milestones(curve, 'v1 5m_vs_6m (5M)')
