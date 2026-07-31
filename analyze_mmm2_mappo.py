"""Analyze MAPPO MMM2 baseline."""
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Find all MAPPO MMM2 runs
base_paths = [
    "C:/Users/张英铭/Desktop/5/results/results/results/StarCraft2/MMM2",
    "C:/Users/张英铭/Desktop/on-policy/onpolicy/scripts/results/StarCraft2/MMM2",
]

mappo_runs = []
for base in base_paths:
    if not os.path.exists(base):
        continue
    for root, dirs, files in os.walk(base):
        if 'eval_win_rate' in dirs:
            inner = os.path.join(root, 'eval_win_rate', 'eval_win_rate')
            if os.path.exists(inner):
                rel = os.path.relpath(root, base)
                mappo_runs.append((base, rel, inner))

print(f"Found {len(mappo_runs)} MMM2 runs:")
for base, rel, _ in mappo_runs:
    print(f"  {base}/{rel}")
print()

# Extract eval_win_rate for each
for base, rel, log_dir in mappo_runs:
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        if 'eval_win_rate' in ea.Tags()['scalars']:
            events = ea.Scalars('eval_win_rate')
            print(f"=== {base}/{rel} ===")
            print(f"  Total points: {len(events)}")
            if events:
                print(f"  First step: {events[0].step:,}, last step: {events[-1].step:,}")
                win_rates = [e.value for e in events]
                print(f"  Peak: {max(win_rates):.2%}")
                print(f"  Final: {win_rates[-1]:.2%}")
                # Find first non-zero
                first_nonzero = next((e.step for e in events if e.value > 0), None)
                print(f"  First non-zero win: step {first_nonzero}")
                # Sample at milestones
                milestones = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
                print(f"  Win rate at milestones:")
                for ms in milestones:
                    closest = min(events, key=lambda e: abs(e.step - ms))
                    if abs(closest.step - ms) < 100_000:
                        print(f"    step {closest.step:>10,}: {closest.value:.2%}")
            print()
    except Exception as e:
        print(f"Error for {rel}: {e}")
