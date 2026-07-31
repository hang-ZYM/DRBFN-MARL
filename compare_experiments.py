"""Compare eval win rates across all 3m and 5m_vs_6m experiments."""
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_winrate_curve(log_dir):
    """Returns list of (step, win_rate) tuples."""
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        if 'eval_win_rate' in ea.Tags()['scalars']:
            events = ea.Scalars('eval_win_rate')
            return [(e.step, e.value) for e in events]
    except Exception as e:
        return f'Error: {e}'
    return []

def summarize(curve, milestones=None):
    """Print summary stats for a curve."""
    if isinstance(curve, str):
        return curve
    if not curve:
        return 'no data'
    steps = [c[0] for c in curve]
    winrates = [c[1] for c in curve]
    final = winrates[-1]
    peak = max(winrates)
    peak_step = steps[winrates.index(peak)]

    # Mean of last 10% (mean10)
    n = len(winrates)
    last_10pct = winrates[max(0, n - max(5, n // 10)):]
    mean_last = sum(last_10pct) / len(last_10pct)

    # First milestone >= 50%, 90%
    first_50 = next((s for s, w in curve if w >= 0.5), None)
    first_90 = next((s for s, w in curve if w >= 0.9), None)

    return {
        'final_step': steps[-1],
        'final': final,
        'peak': peak,
        'peak_step': peak_step,
        'mean_last': mean_last,
        'first_50_step': first_50,
        'first_90_step': first_90,
    }

def find_experiments(map_name):
    """Find all eval_win_rate logs for a map."""
    base = f'C:/Users/张英铭/Desktop/on-policy/onpolicy/scripts/results/StarCraft2/{map_name}'
    results = []
    for root, dirs, files in os.walk(base):
        if 'eval_win_rate' in dirs:
            inner = os.path.join(root, 'eval_win_rate', 'eval_win_rate')
            if os.path.exists(inner):
                # Extract a friendly name
                rel = os.path.relpath(root, base)
                results.append((rel, inner))
    return results

print('=' * 80)
print(' 3m Experiments Comparison')
print('=' * 80)
experiments = find_experiments('3m')
print(f'\nFound {len(experiments)} experiments:\n')

for name, log_dir in sorted(experiments):
    curve = get_winrate_curve(log_dir)
    s = summarize(curve)
    if isinstance(s, str):
        print(f'  {name}: {s}')
    else:
        print(f'  {name}')
        print(f'    steps: {s["final_step"]:>8} | final: {s["final"]:.2%} | peak: {s["peak"]:.2%} @ step {s["peak_step"]}')
        print(f'    mean_last: {s["mean_last"]:.2%} | first≥50%: step {s["first_50_step"]} | first≥90%: step {s["first_90_step"]}')

print('\n')
print('=' * 80)
print(' 5m_vs_6m Experiments Comparison')
print('=' * 80)
experiments = find_experiments('5m_vs_6m')
print(f'\nFound {len(experiments)} experiments:\n')

for name, log_dir in sorted(experiments):
    curve = get_winrate_curve(log_dir)
    s = summarize(curve)
    if isinstance(s, str):
        print(f'  {name}: {s}')
    else:
        print(f'  {name}')
        print(f'    steps: {s["final_step"]:>8} | final: {s["final"]:.2%} | peak: {s["peak"]:.2%} @ step {s["peak_step"]}')
        print(f'    mean_last: {s["mean_last"]:.2%} | first≥50%: step {s["first_50_step"]} | first≥90%: step {s["first_90_step"]}')
