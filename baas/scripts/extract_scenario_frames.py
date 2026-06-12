"""Frame-level NPZ extractor for scenario_frames/.

Separate from the BAAS paper dataset release (runs/catalogue.json + release/).
Replays each catalogue scenario via the same deterministic path as
replay_by_scenario_id (resolve_replay_inputs) and writes one
{scenario_id}.npz per scenario with per-frame RGB, kinematic state, actions,
reward, done and critical-incident arrays, plus scalar metadata.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from baas.adapters.highway_env.adapter import HighwayEnvAdapter
from baas.core.ego_policy import DQNEgoPolicy
from baas.core.rollout import run_episode
from baas.evaluation.replay import resolve_replay_inputs

logger = logging.getLogger(__name__)


def extract_scenario(scenario_id, catalogue_path, adapter, ego_policy, device="cpu"):
    row, spec, adv_controllers, post_reset_fn, env_cfg, thresholds = resolve_replay_inputs(
        scenario_id, catalogue_path, adapter, device=device,
    )

    result = run_episode(
        spec, adapter, ego_policy, adv_controllers, thresholds,
        env_cfg=env_cfg, render_mode="rgb_array", record_frames=True,
        stop_on_critical=False, post_reset_fn=post_reset_fn,
    )

    T = result.ego_trace.shape[0]
    rgb = np.stack(result.frames[:T], axis=0).astype(np.uint8)

    if result.adv_traces:
        adv_states = np.stack(result.adv_traces, axis=1).astype(np.float32)
    else:
        adv_states = np.zeros((T, 0, 4), dtype=np.float32)

    done = np.zeros(T, dtype=bool)
    if T > 0:
        done[-1] = True

    if result.metrics.critical_incident:
        critical_incident = np.arange(1, T + 1) >= result.metrics.time_to_critical_incident_steps
    else:
        critical_incident = np.zeros(T, dtype=bool)

    feasibility = row.get("feasibility")
    feasibility = float(feasibility) if feasibility is not None else float("nan")

    return dict(
        rgb=rgb,
        ego_state=result.ego_trace,
        adv_states=adv_states,
        ego_action=result.ego_actions,
        adv_actions=result.adv_actions,
        reward=result.rewards,
        done=done,
        critical_incident=critical_incident,
        scenario_id=row["scenario_id"],
        method=row["method"],
        env_seed=int(row["env_seed"]),
        difficulty_label=row.get("difficulty_label") or "",
        feasibility=feasibility,
        dt=float(1.0 / env_cfg.policy_frequency),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogue", type=Path, default=Path("runs/catalogue.json"))
    ap.add_argument("--ego-policy", type=Path, default=Path("pretrained/frozen_model_dqn_cnn.zip"))
    ap.add_argument("--output", type=Path, default=Path("scenario_frames"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--scenario-id", default=None,
        help="Extract a single scenario_id only (for smoke-testing the output structure).",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    catalogue = json.loads(args.catalogue.read_text(encoding="utf-8"))
    scenario_ids = [args.scenario_id] if args.scenario_id else [r["scenario_id"] for r in catalogue]

    adapter = HighwayEnvAdapter()
    ego_policy = DQNEgoPolicy(str(args.ego_policy), device=args.device)

    args.output.mkdir(parents=True, exist_ok=True)

    for i, scenario_id in enumerate(scenario_ids, 1):
        out_path = args.output / f"{scenario_id}.npz"
        if out_path.exists():
            logger.info("[%d/%d] %s (skip, exists)", i, len(scenario_ids), scenario_id)
            continue
        logger.info("[%d/%d] %s", i, len(scenario_ids), scenario_id)
        data = extract_scenario(scenario_id, args.catalogue, adapter, ego_policy, device=args.device)
        np.savez(out_path, **data)


if __name__ == "__main__":
    main()
