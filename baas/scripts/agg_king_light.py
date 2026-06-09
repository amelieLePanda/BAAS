"""Print per-seed KING-light stats and mean±std across seeds 0–4."""
import json
import numpy as np
from pathlib import Path
from baas.evaluation.summarise import summarise_runs, summarise_multiseed

BASE = Path("runs/king_light")
per_seed = []
print(f"{'seed':>5}  {'p_coll':>7}  {'TTCI_adv':>9}  {'feasibility':>12}  {'max_jerk_mean':>14}")
print("-" * 60)
for seed in range(5):
    p = BASE / f"run_king_light_seed{seed}" / "results.json"
    if not p.exists():
        print(f"  {seed}  MISSING")
        continue
    data = json.loads(p.read_text())
    s = summarise_runs([data])
    key = next(iter(s))
    st = s[key]
    per_seed.append(s)
    jerks = [r["metrics"]["max_adv_jerk"] for r in data["rollouts"]
             if r["metrics"].get("max_adv_jerk") is not None]
    print(f"  {seed}  {st.get('p_collision', float('nan')):>7.3f}  "
          f"{st.get('mean_ttci_adv_steps') or float('nan'):>9.1f}  "
          f"{st.get('mean_feasibility') or float('nan'):>12.3f}  "
          f"{np.mean(jerks) if jerks else float('nan'):>14.1f}")

if len(per_seed) == 5:
    m = summarise_multiseed(per_seed)[next(iter(summarise_multiseed(per_seed)))]
    print(f"\n  mean  {m.get('p_collision_mean', float('nan')):>7.3f}  "
          f"{m.get('mean_ttci_adv_steps_mean') or float('nan'):>9.1f}  "
          f"{m.get('mean_feasibility_mean') or float('nan'):>12.3f}")
    print(f"  std   {m.get('p_collision_std', float('nan')):>7.3f}  "
          f"{m.get('mean_ttci_adv_steps_std') or float('nan'):>9.1f}  "
          f"{m.get('mean_feasibility_std') or float('nan'):>12.3f}")
else:
    print(f"\nOnly {len(per_seed)}/5 seeds found. Run missing seeds first.")
