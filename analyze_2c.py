"""Compare 2c_vs_64zg baselines."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

paths = [
    ("MAPPO (cloud)", "C:/Users/张英铭/Desktop/5/results/results/results/StarCraft2/2c_vs_64zg/rmappo/check/run2/logs/eval_win_rate/eval_win_rate"),
    ("v1 (local)", "C:/Users/张英铭/Desktop/on-policy/onpolicy/scripts/results/StarCraft2/2c_vs_64zg/r_drbfn/drbfn_2c_vs_64zg/run2/logs/eval_win_rate/eval_win_rate"),
]

for name, log in paths:
    try:
        ea = EventAccumulator(log)
        ea.Reload()
        events = ea.Scalars('eval_win_rate')
        ws = [e.value for e in events]
        print(f"=== {name} ===")
        print(f"  Steps: {events[0].step:,} - {events[-1].step:,}, total points: {len(events)}")
        print(f"  Peak: {max(ws):.2%}")
        print(f"  Final: {ws[-1]:.2%}")
        first_nonzero = next((e.step for e in events if e.value > 0), None)
        print(f"  First non-zero: step {first_nonzero}")
        for ms in [50_000, 100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]:
            closest = min(events, key=lambda e: abs(e.step - ms))
            if abs(closest.step - ms) < 50_000:
                print(f"    step {closest.step:>10,}: {closest.value:.2%}")
        print()
    except Exception as e:
        print(f"{name}: {e}")
