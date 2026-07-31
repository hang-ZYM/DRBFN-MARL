"""Final results analysis of 5m_vs_6m 5M QVPO run."""
import re

LOG = "C:/Users/张英铭/Desktop/on-policy/results/qvpo_5m6m_5M_20260729_203307/log.txt"

# Parse all eval win rates
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

print(f"Total evals: {len(eval_data)}")
print()

# All evals
print("=" * 60)
print(" ALL EVAL WIN RATES")
print("=" * 60)
print(f"{'#':>3} | {'update':>6} | {'win_rate':>10}")
print("-" * 28)
for i, (u, w) in enumerate(eval_data):
    marker = ' PEAK' if w >= 0.85 else (' LOW' if w < 0.5 else '')
    print(f"{i+1:>3} | {u:>6} | {w:.2%}{marker}")

# Stats
win_rates = [w for _, w in eval_data]
print()
print("=" * 60)
print(" FINAL STATS")
print("=" * 60)
print(f"Total evals:       {len(win_rates)}")
print(f"Initial (warmup):  {win_rates[0]:.2%}")
print(f"Peak:              {max(win_rates):.2%} @ update {eval_data[win_rates.index(max(win_rates))][0]}")
print(f"Final:             {win_rates[-1]:.2%}")
print(f"Times >=80%:       {sum(1 for w in win_rates if w >= 0.8)}")
print(f"Times >=70%:       {sum(1 for w in win_rates if w >= 0.7)}")
print(f"Times <50%:        {sum(1 for w in win_rates if w < 0.5)}")
print()
print(f"Mean of last 5:    {sum(win_rates[-5:])/5:.2%}")
print(f"Mean of last 10:   {sum(win_rates[-10:])/10:.2%}")
print(f"Mean of last 20:   {sum(win_rates[-20:])/20:.2%}")
print(f"Mean of all:       {sum(win_rates)/len(win_rates):.2%}")
