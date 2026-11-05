"""Closed-loop drawer eval for a LeRobot position policy (ACT / SmolVLA).

Runs on halo with EGL. Loads a pretrained policy and steps absolute joint
actions on ``RaxDrawerEnv`` with the place success contract.

  MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.vla.eval_policy \\
    --policy ./outputs/train/rax_drawer_act_state6/checkpoints/030000/pretrained_model \\
    --episodes 20 --split test

GUI (realtime, local display):

  DISPLAY=:0 MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.vla.eval_policy \\
    --policy ./outputs/train/rax_drawer_act_state6/checkpoints/030000/pretrained_model \\
    --episodes 1 --split explore --gui
"""

from __future__ import annotations

import argparse
import json
import time
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


def _slice_obs_state_stats(pipeline, state_dim: int) -> None:
    """Checkpoints may keep 15-D dataset stats while the policy state is 6-D."""
    if pipeline is None or state_dim <= 0:
        return
    for step in getattr(pipeline, "steps", []):
        stats = getattr(step, "stats", None)
        if not isinstance(stats, dict) or "observation.state" not in stats:
            continue
        block = stats["observation.state"]
        if not isinstance(block, dict):
            continue
        for key, value in list(block.items()):
            if isinstance(value, torch.Tensor) and value.shape[-1] > state_dim:
                block[key] = value[..., :state_dim].contiguous()
            else:
                arr = np.asarray(value)
                if arr.ndim >= 1 and arr.shape[-1] > state_dim:
                    block[key] = arr[..., :state_dim]


def _load_policy(path: Path, device: str, state_dim: int):
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(path)
    policy.to(device)
    policy.eval()
    # Must pass pretrained_path so MEAN_STD normalizer/unnormalizer stats load from
    # the checkpoint. dataset_stats=None alone builds empty stats → identity I/O and
    # joint commands stay in normalized space (observed ACT open-loop MSE ~0.5 vs ~4e-4).
    preprocess, postprocess = make_pre_post_processors(
        cfg,
        pretrained_path=str(path),
        dataset_stats=None,
    )
    feat = getattr(cfg, "input_features", {}).get("observation.state")
    cfg_dim = int(feat.shape[0]) if feat is not None else state_dim
    _slice_obs_state_stats(preprocess, min(state_dim, cfg_dim))
    return policy, preprocess, postprocess, cfg


def _obs_batch(
    env: RaxDrawerEnv,
    *,
    state_dim: int,
    device: str,
    rename: dict[str, str],
    image_keys: list[str] | None = None,
):
    from labs.rax_drawer.vla.depth_vis import DEPTH_CAM_KEY, depth_m_to_rgb3

    raw = env.robot.get_observation()
    state = np.array([float(raw[f"{n}.pos"]) for n in JOINTS[:state_dim]], dtype=np.float32)
    batch = {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "task": ["Open the drawer, place the green cube inside, and close the drawer"],
    }
    wanted = set(image_keys) if image_keys is not None else None
    for cam in env.task.camera_names:
        key = f"observation.images.camera_{cam}"
        if wanted is not None and key not in wanted and rename.get(key, key) not in wanted:
            continue
        img = raw[f"camera_{cam}"]
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        batch[rename.get(key, key)] = t
    depth_key = f"observation.images.{DEPTH_CAM_KEY}"
    depth_mapped = rename.get(depth_key, depth_key)
    if wanted is not None and (depth_key in wanted or depth_mapped in wanted):
        img = depth_m_to_rgb3(env.top_depth_m())
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        batch[depth_mapped] = t
    return batch


def run_episode(
    env: RaxDrawerEnv,
    policy,
    preprocess,
    postprocess,
    *,
    state_dim: int,
    device: str,
    rename,
    max_s: float,
    realtime: bool = False,
    image_keys: list[str] | None = None,
) -> bool:
    env.reset(include_cameras=False)
    policy.reset()
    fps = float(env.robot.config.record_fps)
    dt = 1.0 / fps
    steps = int(max_s * fps)
    for _ in range(steps):
        t0 = time.perf_counter()
        batch = _obs_batch(
            env, state_dim=state_dim, device=device, rename=rename, image_keys=image_keys
        )
        with torch.inference_mode():
            obs = preprocess(batch) if preprocess is not None else batch
            action = policy.select_action(obs)
            if postprocess is not None:
                action = postprocess(action)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action).reshape(-1)
        cmd = {f"{JOINTS[i]}.pos": float(action[i]) for i in range(min(len(JOINTS), action.shape[0]))}
        result = env.step(cmd, duration_s=dt, include_cameras=False)
        if realtime:
            # Pace to wall-clock so the GLFW window is watchable.
            lag = dt - (time.perf_counter() - t0)
            if lag > 0:
                time.sleep(lag)
            window = getattr(env.robot, "_glfw_window", None)
            if window is not None:
                import glfw

                if glfw.window_should_close(window):
                    print("GUI window closed — stopping episode", flush=True)
                    return bool(env.check_success())
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
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the MuJoCo GLFW window and pace at realtime (~30 Hz).",
    )
    parser.add_argument(
        "--gui-top-depth",
        action="store_true",
        help="With --gui, show colorized top-camera depth in the top-right pane.",
    )
    parser.add_argument(
        "--hold-s",
        type=float,
        default=3.0,
        help="With --gui, keep the final pose visible this many seconds after each episode.",
    )
    args = parser.parse_args()
    rename = json.loads(args.rename_map) if args.rename_map else {}
    if args.gui_top_depth and not args.gui:
        raise SystemExit("--gui-top-depth requires --gui")

    manifests = Path(__file__).resolve().parents[1] / "manifests" / f"{args.split}.json"
    scenes = json.loads(manifests.read_text())["scenes"][: args.episodes]
    policy, preprocess, postprocess, cfg = _load_policy(
        args.policy, args.device, state_dim=args.state_dim
    )
    image_keys = [
        key
        for key, ft in getattr(cfg, "input_features", {}).items()
        if key.startswith("observation.images.")
    ]
    print(f"policy image keys: {image_keys}", flush=True)

    task = DrawerTaskConfig(
        require_cube_placed=True,
        episode_limit_s=args.max_s + 5.0,
        render_window=bool(args.gui),
        gui_show_top_depth=bool(args.gui_top_depth),
    )
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
    if env._depth_renderer is not None:
        env._depth_renderer.close()
        h, w = env.robot._camera_resolution("top")
        env._depth_renderer = mj.Renderer(env.model, height=h, width=w)
        env._depth_renderer.enable_depth_rendering()

    if args.gui:
        if env.robot._glfw_window is None:
            raise SystemExit("GUI requested but GLFW window was not created (check DISPLAY/XAUTHORITY)")
        print("GUI window open — watch SO-101 MuJoCo Recording", flush=True)

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
                realtime=bool(args.gui),
                image_keys=image_keys,
            )
            err = ""
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = f"{type(exc).__name__}: {exc}"
        results.append({"i": i, "success": ok, "error": err})
        print(f"eval {i}: success={ok} {err}", flush=True)
        if args.gui and args.hold_s > 0 and env.robot._glfw_window is not None:
            import glfw

            t_end = time.time() + args.hold_s
            while time.time() < t_end and not glfw.window_should_close(env.robot._glfw_window):
                env.robot._render_glfw()
                time.sleep(0.03)

    env.close()
    ok_n = sum(1 for r in results if r["success"])
    out = {"n": len(results), "successes": ok_n, "rate": ok_n / max(len(results), 1), "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"successes": ok_n, "n": len(results), "out": str(args.out)}))


if __name__ == "__main__":
    main()
