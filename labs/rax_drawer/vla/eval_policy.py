"""Closed-loop drawer eval for a LeRobot position policy (ACT / SmolVLA).

Runs on halo with EGL. Loads a pretrained policy and steps absolute joint
actions on ``RaxDrawerEnv`` with the place success contract.

  MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.vla.eval_policy \\
    --policy ./outputs/train/rax_drawer_act_state6/checkpoints/030000/pretrained_model \\
    --episodes 20 --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from labs.rax_drawer.environment import DrawerTaskConfig, RaxDrawerEnv

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def _load_policy(path: Path, device: str):
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(path)
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(cfg, dataset_stats=None)
    return policy, preprocess, postprocess, cfg


def _obs_batch(env: RaxDrawerEnv, *, state_dim: int, device: str, rename: dict[str, str]):
    raw = env.robot.get_observation()
    state = np.array([float(raw[f"{n}.pos"]) for n in JOINTS[:state_dim]], dtype=np.float32)
    batch = {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": ["Open the drawer, place the green cube inside, and close the drawer"],
    }
    for cam in env.task.camera_names:
        key = f"observation.images.camera_{cam}"
        img = raw[f"camera_{cam}"]
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        batch[rename.get(key, key)] = t
    return batch


def run_episode(env: RaxDrawerEnv, policy, preprocess, postprocess, *, state_dim: int, device: str, rename, max_s: float) -> bool:
    env.reset(include_cameras=False)
    policy.reset()
    fps = float(env.robot.config.record_fps)
    steps = int(max_s * fps)
    for _ in range(steps):
        batch = _obs_batch(env, state_dim=state_dim, device=device, rename=rename)
        with torch.inference_mode():
            obs = preprocess(batch) if preprocess is not None else batch
            action = policy.select_action(obs)
            if postprocess is not None:
                action = postprocess(action)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action).reshape(-1)
        cmd = {f"{JOINTS[i]}.pos": float(action[i]) for i in range(min(len(JOINTS), action.shape[0]))}
        result = env.step(cmd, duration_s=1.0 / fps, include_cameras=False)
        if result.get("terminated"):
            return bool(result.get("success"))
    return bool(env.check_success())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--split", choices=("explore", "test"), default="test")
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--max-s", type=float, default=120.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rename-map",
        default="",
        help='JSON map for SmolVLA, e.g. {"observation.images.camera_top":"observation.images.camera1",...}',
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/eval/rax_drawer_eval.json"))
    args = parser.parse_args()
    rename = json.loads(args.rename_map) if args.rename_map else {}

    manifests = Path(__file__).resolve().parents[1] / "manifests" / f"{args.split}.json"
    scenes = json.loads(manifests.read_text())["scenes"][: args.episodes]
    policy, preprocess, postprocess, _cfg = _load_policy(args.policy, args.device)

    task = DrawerTaskConfig(require_cube_placed=True, episode_limit_s=args.max_s + 5.0)
    # Match collection resolution.
    env = RaxDrawerEnv(task)
    env.robot.config.camera_resolutions = {
        "top": (640, 480),
        "side": (640, 480),
        "wrist": (640, 480),
    }
    import mujoco as mj

    for name in list(env.robot._renderers):
        env.robot._renderers[name].close()
        del env.robot._renderers[name]
    for name in env.task.camera_names:
        h, w = env.robot._camera_resolution(name)
        env.robot._renderers[name] = mj.Renderer(env.model, height=h, width=w)

    results = []
    for i, scene in enumerate(scenes):
        env.set_layout(x_m=scene["x_m"], y_m=scene["y_m"], yaw_deg=scene["yaw_deg"])
        try:
            ok = run_episode(
                env,
                policy,
                preprocess,
                postprocess,
                state_dim=args.state_dim,
                device=args.device,
                rename=rename,
                max_s=args.max_s,
            )
            err = ""
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = f"{type(exc).__name__}: {exc}"
        results.append({"i": i, "success": ok, "error": err})
        print(f"eval {i}: success={ok} {err}", flush=True)

    env.close()
    ok_n = sum(1 for r in results if r["success"])
    out = {"n": len(results), "successes": ok_n, "rate": ok_n / max(len(results), 1), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"successes": ok_n, "n": len(results), "out": str(args.out)}))


if __name__ == "__main__":
    main()
