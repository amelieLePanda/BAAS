# BAAS: Benchmarking Adversarial Agent Strategies

> **DRAFT.** Results for QD-RL are pending a full rerun. This README will be updated once the final experiment is complete.

A reproducible benchmarking framework for adversarial scenario generation in autonomous driving validation. Five adversarial paradigms are compared under identical conditions against a frozen ego policy in a highway driving environment.

**Paper:** C. M.-L. Frischknecht-Gruber, M. Reif, A. Fischer. *BAAS: Benchmarking Adversarial Agent Strategies. A Comparative Study of Gradient-based, Reinforcement, and Evolutionary Paradigms for Safety-Critical Scenario Generation.* Proc. 36th European Safety and Reliability Conference (ESREL 2026)

---

## Methods

| Method | Paradigm | Description |
|---|---|---|
| Parameter Sweep | Non-learning baseline | Deterministic grid search over initial adversary positions and velocities |
| KING-light | Gradient-based | Differentiable bicycle proxy that optimises a continuous action sequence via gradient descent |
| PPO Adversary | Reinforcement learning | Single adversary trained with PPO against the frozen ego |
| MAP-Elites | Quality-Diversity | Open-loop action sequences evolved to fill a behavioural descriptor archive |
| QD-RL | QD + Reinforcement learning | MAP-Elites archive of PPO policies, each trained with different reward shaping |

---

## Results

Results below are from extended experiment runs (May–June 2026), superseding the published paper values. See the **Corrections** section for details on numerical differences.

### Effectiveness

| Method | p_coll | Mean TTCI_adv (steps) |
|---|---|---|
| Parameter Sweep | 0.633 | - |
| KING-light | 0.433 | 114.9 |
| PPO Adversary | 0.633 | 78.2 |
| MAP-Elites | 0.500 | 140.4 |
| QD-RL | *pending* | *pending* |

### Feasibility and Plausibility

| Method | Feasibility | Max Adv Jerk (m/s³) |
|---|---|---|
| Parameter Sweep | - | - |
| KING-light | 0.570 | 43.3 |
| PPO Adversary | 0.263 | 53.9 |
| MAP-Elites | 0.593 | 38.3 |
| QD-RL | *pending* | *pending* |

Feasibility = fraction of fixed-perturbation reruns where the ego avoids terminal failure. Higher feasibility means the scenario is challenging but solvable, which is the target property for V&V-relevant scenarios.

### Diversity in φ-space (evaluation rollouts)

| Method | Coverage (φ) | Entropy (φ) | Diversity Score |
|---|---|---|---|
| Parameter Sweep | - | - | - |
| KING-light | 0.140 | 0.528 | 0.334 |
| PPO Adversary | 0.110 | 0.470 | 0.290 |
| MAP-Elites | 0.090 | 0.423 | 0.257 |
| QD-RL | *pending* | *pending* | *pending* |

---

> **Note on Results (Update to Published Paper)**
> The extended runs (May–June 2026) supersede the values in the ESREL 2026 paper. Two systematic differences exist. First, jerk is now computed from kinematic replay traces at `dt = 0.5 s`, giving physically coherent values (16–54 m/s³). The paper's values (up to 1676 m/s³) were from an earlier implementation and are physically implausible. Second, QD-RL results are pending a full rerun under the revised configuration (10×10 archive, burst\_steps=25 000, ent\_coef=0.02). The paper's QD-RL feasibility of 0.767 was from a run that terminated early. All other method results are from complete runs.

> **Note on Feasibility Across Method Types**
> Feasibility is estimated by replaying the same adversary 10 times with varied ego environment seeds and counting the fraction of reruns in which the ego survives. For active adversary methods (KING-light, PPO, MAP-Elites, QD-RL) the adversary controller is held fixed across reruns. For the parameter sweep, the background IDM/MOBIL vehicle is repositioned to the worst-case initial conditions found during the sweep. Feasibility there reflects how reliably that initial configuration leads to a critical outcome, not the robustness of a trained policy. The two feasibility values are comparable in direction (higher means harder but solvable) but differ structurally: parameter sweep feasibility characterises a static perturbation, whereas active-method feasibility characterises a trained adversarial behaviour.

---

## Example Scenarios

*GIFs will be added here once selected from the final evaluation runs.*

---

## Dataset Release

The scenario catalogue (`runs/catalogue.json`) records every evaluated rollout across all methods. It is a flat JSON array, one object per rollout.

### Catalogue schema

| Field | Type | Description |
|---|---|---|
| `scenario_id` | str | Unique identifier: `{method}_s{opt_seed}_{rollout_index:03d}` |
| `method` | str | `parameter_sweep`, `king_light`, `ppo_adversary`, `map_elites`, `qdrl` |
| `opt_seed` | int | Seed index used during optimisation (0 for single-run methods) |
| `rollout_index` | int | Index into the shared rollout spec pool |
| `env_seed` | int | Gymnasium environment seed |
| `critical_incident` | bool | Whether a critical incident was triggered |
| `ego_collision` | bool | Whether the ego vehicle collided |
| `feasibility` | float\|null | Fraction of 10 reruns in which ego survives |
| `difficulty_label` | str\|null | Tier based on feasibility (see below) |
| `phi_dist` | float\|null | Behavioural descriptor: `min_dist_ego_adv` in metres |
| `phi_ttci` | float\|null | Behavioural descriptor: TTCI_adv in steps |
| `min_dist_ego_adv` | float\|null | Minimum ego-to-adversary distance in metres |
| `ttci_adv_steps` | int\|null | Time to critical incident involving adversary (steps) |
| `max_adv_jerk` | float\|null | Maximum adversary jerk in m/s³ |
| `results_path` | str | Path to `results.json` relative to catalogue dir |
| `artefact_reference` | str | Path to adversary artefact relative to catalogue dir, or `""` |

### Difficulty labels

| Label | Feasibility range | Interpretation |
|---|---|---|
| `degenerate` | < 0.10 | Ego almost always fails regardless of adversary |
| `hardcore` | 0.10 to 0.30 | Ego rarely escapes |
| `hard` | 0.30 to 0.50 | Ego escapes less than half the time |
| `medium` | 0.50 to 0.70 | Adversary holds an edge |
| `easy` | 0.70 to 1.00 | Ego mostly survives |
| `trivial` | 1.00 | Ego always survives |

### Generating or extending the catalogue

```bash
# Add a method's run(s) to the catalogue (append-safe, skips duplicate scenario_ids):
python -m baas.data.catalogue \
    --catalogue runs/catalogue.json \
    --run-dirs runs/map_elites/run_000 \
    --artefact-name archive.json \
    --opt-seeds 0

# Multi-seed methods (e.g. KING-light seeds 0-4):
python -m baas.data.catalogue \
    --catalogue runs/catalogue.json \
    --run-dirs runs/king_light/run_king_light_seed{0,1,2,3,4} \
    --artefact-name king_light_artefact.json \
    --opt-seeds 0 1 2 3 4
```

### Replaying a scenario by ID

```python
from pathlib import Path
from baas.adapters.highway_env.adapter import HighwayEnvAdapter
from baas.core.ego_policy import DQNEgoPolicy
from baas.core.metrics import IncidentThresholds
from baas.evaluation.replay import replay_by_scenario_id

result = replay_by_scenario_id(
    scenario_id="map_elites_s0_007",
    catalogue_path=Path("runs/catalogue.json"),
    adapter=HighwayEnvAdapter(),
    ego_policy=DQNEgoPolicy("pretrained/frozen_model_dqn_cnn.zip"),
    thresholds=IncidentThresholds(...),
    output_path=Path("replay_map_elites_s0_007.gif"),
)
```

---

## Corrections to Published Paper

The extended runs revealed two systematic differences from the paper's reported values:

**Jerk values**: The paper reported Max Adv Jerk values of 1676 m/s³ (KING-light) and 961 m/s³ (MAP-Elites). The current implementation computes jerk from kinematic replay traces using `dt = 1/policy_frequency = 0.5 s`, yielding physically coherent values in the 16–54 m/s³ range. The paper's values were produced by an earlier implementation and are physically implausible for highway vehicles (comfort limit ~6 m/s³, emergency limit ~60–100 m/s³). The conceptual ordering is preserved: QD-RL still produces the lowest adversary jerk of all active methods.

**QD-RL feasibility**: The paper reported QD-RL feasibility of 0.767, based on a run that terminated early. The new run uses an improved configuration (10×10 archive, burst_steps=25 000, n_envs=4, ent_coef=0.02) and will report updated values once complete.

**PPO terminal reward**: The paper defines the terminal reward on ego collision. In the implementation it fires on any critical incident, including proximity-only triggers. This is deliberate and produces a slightly more aggressive adversary during training. It does not affect evaluation metrics, which use the shared neutral evaluation path.

---

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Running Experiments

All scripts are in `baas/scripts/`. Each takes `--config configs/benchmark_v1.yaml` and `--ego-policy pretrained/frozen_model_dqn_cnn.zip`.

```bash
# Parameter sweep (baseline)
python baas/scripts/run_parameter_sweep.py \
    --config configs/benchmark_v1.yaml \
    --ego-policy pretrained/frozen_model_dqn_cnn.zip \
    --output runs/parameter_sweep/run_000

# KING-light (gradient-based)
python baas/scripts/run_king_light.py \
    --config configs/benchmark_v1.yaml \
    --ego-policy pretrained/frozen_model_dqn_cnn.zip \
    --output runs/king_light/run_000

# PPO adversary (RL)
python baas/scripts/run_ppo_adversary.py \
    --config configs/benchmark_v1.yaml \
    --ego-policy pretrained/frozen_model_dqn_cnn.zip \
    --output runs/ppo_adversary/run_000

# MAP-Elites (QD)
python baas/scripts/run_map_elites.py \
    --config configs/benchmark_v1.yaml \
    --ego-policy pretrained/frozen_model_dqn_cnn.zip \
    --output runs/map_elites/run_000

# QD-RL
python baas/scripts/run_qdrl.py \
    --config configs/benchmark_v1.yaml \
    --ego-policy pretrained/frozen_model_dqn_cnn.zip \
    --output runs/qdrl/run_000

# Summarise all results
python baas/scripts/run_benchmark.py summarise runs/ --output runs/summary
```

---

## Cluster / Long-running Jobs

QD-RL is the most compute-intensive method. Recommended settings (already set in `configs/benchmark_v1.yaml`):

```
burst_steps: 25000   # training steps per candidate
n_envs: 4            # parallel environments (use SubprocVecEnv)
archive_grid_dims: [10, 10]
n_iters: 80
ent_coef: 0.02
device: auto         # set to "cuda" if GPU is available
```

Expected wall time: ~8–12 hours on a modern CPU node, ~4–6 hours with GPU.

---

## Citation

```bibtex
@inproceedings{frischknecht2026baas,
  title     = {{BAAS}: Benchmarking Adversarial Agent Strategies ---
               A Comparative Study of Gradient-based, Reinforcement,
               and Evolutionary Paradigms for Safety-Critical Scenario Generation},
  author    = {Frischknecht-Gruber, Carmen Mei-Ling and Reif, Monika and Fischer, Andreas},
  booktitle = {Proceedings of the 36th European Safety and Reliability Conference (ESREL 2026)},
  year      = {2026},
  publisher = {Research Publishing}
}
```
