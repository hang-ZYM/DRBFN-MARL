"""List ALL QVPO experiments with key metrics."""
import os
import re
import glob

LOGS_DIR = "C:/Users/张英铭/Desktop/on-policy/results"

# Find all qvpo log directories
all_logs = []
for d in sorted(os.listdir(LOGS_DIR)):
    if 'qvpo' in d.lower():
        log_path = os.path.join(LOGS_DIR, d, 'log.txt')
        if os.path.exists(log_path):
            all_logs.append((d, log_path))

print(f"Found {len(all_logs)} QVPO experiment logs\n")

step_pattern = re.compile(
    r"\[v_qvpo step=(\d+)\].*?qtot_loss=([-\d.]+).*?drbfn_loss=([-\d.]+).*?"
    r"phi_scale=([-\d.]+).*?g_i_scale=([-\d.]+).*?"
    r"log_p_mean=([-\d.]+).*?grad_norm=([-\d.]+)"
)
eval_pattern = re.compile(r"eval win rate is ([\d.]+)\.")
update_pattern = re.compile(r"updates (\d+)/")

for dirname, log_path in all_logs:
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Get start/end times
    times = re.findall(r"(Started at|Finished at): (.*)", content)

    # Get all step data
    steps = []
    for m in step_pattern.finditer(content):
        steps.append({
            'step': int(m.group(1)),
            'qtot_loss': float(m.group(2)),
            'drbfn_loss': float(m.group(3)),
            'phi_scale': float(m.group(4)),
            'g_i_scale': float(m.group(5)),
            'log_p_mean': float(m.group(6)),
            'grad_norm': float(m.group(7)),
        })

    # Get all eval win rates
    evals = []
    current_update = 0
    for line in content.split('\n'):
        m = update_pattern.search(line)
        if m:
            current_update = int(m.group(1))
        m = eval_pattern.search(line)
        if m:
            evals.append((current_update, float(m.group(1))))

    # Find clamp value from command line
    clamp_match = re.search(r"--drbfn_phi_clamp\s+([\d.]+)", content)
    clamp = clamp_match.group(1) if clamp_match else "(default)"

    # Find map
    map_match = re.search(r"--map_name\s+(\S+)", content)
    map_name = map_match.group(1) if map_match else "?"

    # Find num_env_steps
    steps_match = re.search(r"--num_env_steps\s+(\d+)", content)
    total_steps_planned = int(steps_match.group(1)) if steps_match else 0

    # Get last update
    last_update_match = None
    for m in update_pattern.finditer(content):
        last_update_match = m
    last_update = int(last_update_match.group(1)) if last_update_match else 0

    # Stats
    n_steps_data = len(steps)
    n_evals = len(evals)
    peak_winrate = max((w for _, w in evals), default=0)
    final_winrate = evals[-1][1] if evals else 0
    max_phi = max((s['phi_scale'] for s in steps), default=0)
    max_grad = max((s['grad_norm'] for s in steps), default=0)
    explosions = sum(1 for s in steps if s['grad_norm'] > 100)

    print(f"=== {dirname} ===")
    print(f"  Map: {map_name} | clamp: {clamp} | planned: {total_steps_planned/1e6:.1f}M")
    print(f"  Last update: {last_update} | step data pts: {n_steps_data} | eval pts: {n_evals}")
    print(f"  Peak winrate: {peak_winrate:.2%} | Final: {final_winrate:.2%}")
    print(f"  Max phi_scale: {max_phi:.4f} | Max grad_norm: {max_grad:.2f} | Explosions: {explosions}")
    if times:
        for label, t in times:
            print(f"  {label}: {t}")
    if n_steps_data > 0:
        last = steps[-1]
        print(f"  Last metrics: phi={last['phi_scale']:.3f}, log_p={last['log_p_mean']:.2f}, g_i={last['g_i_scale']:.3f}")
    print()
