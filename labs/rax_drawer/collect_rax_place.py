"""Roll out the frozen RAX place controller into a LeRobot v3 dataset.

Runs on a host with usable MuJoCo GL (halo / Radeon 8060S + EGL). Successful
episodes only by default. Absolute joint `.pos` actions are logged every control
tick so ACT / SmolVLA can imitate the expert in position mode.

  MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.collect_rax_place --target 100
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

# Import LeRobot before mujoco (OSMesa/torch ordering).
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

from labs.rax_drawer.environment import DrawerTaskConfig, RaxDrawerEnv
from labs.rax_drawer.rax_place.code.open_place_close.controller import main as place_main
from labs.rax_drawer.rax_place.tools import Session
from labs.rax_drawer.vla.depth_vis import DEPTH_CAM_KEY, depth_m_to_rgb3

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
TASK = "Open the drawer, place the green cube inside, and close the drawer"
HERE = Path(__file__).resolve().parent
MANIFESTS = HERE / "manifests"


class PlaceRecorder:
    """Log absolute joint ticks + RGB (+ optional top-depth viz) into LeRobot v3."""

    def __init__(
        self,
        robot,
        *,
        repo_id: str,
        root: Path,
        save_failures: bool = False,
        depth_env: RaxDrawerEnv | None = None,
        record_top_depth: bool = True,
    ):
        self.robot = robot
        self.depth_env = depth_env
        self.record_top_depth = bool(record_top_depth and depth_env is not None)
        self.root = Path(root)
        self.repo_id = repo_id
        self.save_failures = save_failures
        self.spawn_log = self.root / "spawns.jsonl"
        # Force position-mode action schema for BC even if the env drives velocity.
        hw_obs = dict(robot.observation_features)
        if self.record_top_depth:
            # ACT needs matching HxWx3 across cameras; encode metric depth as gray RGB.
            h, w = robot._camera_resolution("top")
            hw_obs[DEPTH_CAM_KEY] = (h, w, 3)
        obs_ft = hw_to_dataset_features(hw_obs, OBS_STR, use_video=True)
        act_ft = hw_to_dataset_features(
            {f"{name}.pos": float for name in JOINTS},
            ACTION,
            use_video=True,
        )
        self.features = {**obs_ft, **act_ft}
        info = self.root / "meta" / "info.json"
        tasks = self.root / "meta" / "tasks.parquet"
        if info.is_file() and tasks.is_file():
            self.dataset = LeRobotDataset.resume(
                repo_id,
                root=self.root,
                tolerance_s=1e-3,
                image_writer_threads=2,
            )
            print(f"resume {repo_id} @ {self.root}", flush=True)
        else:
            # Incomplete first-create (missing tasks.parquet) must be wiped.
            if self.root.exists():
                shutil.rmtree(self.root)
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=int(robot.config.record_fps),
                features=self.features,
                root=self.root,
                robot_type=robot.name,
                use_videos=True,
                image_writer_threads=2,
            )
            print(f"create {repo_id} @ {self.root}", flush=True)
        self._ep_frames = 0
        self._meta: dict[str, Any] = {}
        self.saved = 0
        self.skipped = 0

    def begin(self, meta: dict[str, Any]) -> None:
        self._meta = dict(meta)
        self._ep_frames = 0

    def tick(self) -> None:
        obs = self.robot.get_observation()
        if self.record_top_depth:
            assert self.depth_env is not None
            obs[DEPTH_CAM_KEY] = depth_m_to_rgb3(self.depth_env.top_depth_m())
        action = {f"{name}.pos": float(obs[f"{name}.pos"]) for name in JOINTS}
        obs_frame = build_dataset_frame(self.features, obs, prefix=OBS_STR)
        act_frame = build_dataset_frame(self.features, action, prefix=ACTION)
        self.dataset.add_frame({**obs_frame, **act_frame, "task": TASK})
        self._ep_frames += 1

    def end(self, *, success: bool, episode: int) -> None:
        row = {"episode": episode, "success": success, "frames": self._ep_frames, **self._meta}
        self.root.mkdir(parents=True, exist_ok=True)
        with self.spawn_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if success or self.save_failures:
            if self._ep_frames > 0:
                self.dataset.save_episode()
                self.saved += 1
                print(
                    f"saved ep={episode} frames={self._ep_frames} success={success} total={self.saved}",
                    flush=True,
                )
            else:
                self.skipped += 1
        else:
            self._discard()
            self.skipped += 1
            print(f"discard ep={episode} frames={self._ep_frames}", flush=True)

    def _discard(self) -> None:
        writer = self.dataset.writer
        if writer.episode_buffer is None:
            return
        if writer.image_writer is not None:
            writer._wait_image_writer()
        episode_index = int(np.asarray(writer.episode_buffer["episode_index"]).reshape(-1)[0])
        for key in self.dataset.meta.video_keys:
            img_dir = writer._get_image_file_dir(episode_index, key)
            if img_dir.is_dir():
                shutil.rmtree(img_dir)
        self.dataset.clear_episode_buffer()

    def close(self) -> None:
        if hasattr(self.dataset, "finalize"):
            try:
                self.dataset.finalize()
            except Exception:
                pass
        print(f"recorder done saved={self.saved} discarded={self.skipped} root={self.root}", flush=True)


def _install_step_hook(env: RaxDrawerEnv, recorder: PlaceRecorder) -> None:
    """Record one dataset frame after every physics send_action inside step()."""
    orig = env.step

    def stepped(action, *, duration_s: float = 0.25, include_cameras: bool = True):
        if env.episode < 0:
            raise RuntimeError("call reset() before step()")
        if env.terminated:
            raise RuntimeError("episode has terminated; call reset() before another step")
        duration = float(duration_s)
        if not np.isfinite(duration) or not 0 < duration <= env.task.max_action_duration_s:
            raise ValueError(f"duration_s must be in (0, {env.task.max_action_duration_s}]")
        bounded = env._bound_action(action)
        frames = max(1, round(duration * env.robot.config.record_fps))
        remaining = env.task.episode_limit_s - (env.data.time - env.start_time)
        if remaining <= 0:
            env.terminated = True
            obs = env.observe(include_cameras=include_cameras)
            return {
                "observation": obs,
                "action_applied": bounded,
                "terminated": True,
                "termination_reason": "time_limit",
                "success": False,
            }
        frames = min(frames, max(1, int(np.ceil(remaining * env.robot.config.record_fps))))
        for _ in range(frames):
            env.robot.send_action(bounded)
            recorder.tick()
            env._update_score()
            if env.check_success() or env.data.time - env.start_time >= env.task.episode_limit_s:
                break
        obs = env.observe(include_cameras=include_cameras)
        succeeded = env.check_success()
        reason = (
            "success"
            if succeeded
            else "time_limit"
            if obs["sim_time_s"] >= env.task.episode_limit_s
            else None
        )
        env.terminated = reason is not None
        return {
            "observation": obs,
            "action_applied": bounded,
            "terminated": env.terminated,
            "termination_reason": reason,
            "success": succeeded,
        }

    env.step = stepped  # type: ignore[method-assign]
    # Keep reference so linters know orig was intentional.
    _ = orig


def _count_saved(root: Path) -> int:
    log = root / "spawns.jsonl"
    if not log.is_file():
        return 0
    n = 0
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            if json.loads(line).get("success"):
                n += 1
        except json.JSONDecodeError:
            continue
    return n


def collect(
    *,
    target: int,
    root: Path,
    repo_id: str,
    save_failures: bool,
    max_attempts: int,
    record_top_depth: bool = True,
) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    already = _count_saved(root)
    print(f"target={target} already_success={already} root={root}", flush=True)
    if already >= target:
        return {"saved": already, "attempts": 0, "done": True}

    with Session(
        split="explore",
        episode_limit_s=180.0,
        audit_path=HERE / "rax_place" / "logs" / "mcp_collect.jsonl",
    ) as ctx:
        # Lab-01-like RGB sizes for faster EGL and lighter datasets.
        import mujoco as mj

        ctx.env.robot.config.camera_resolutions = {
            "top": (640, 480),
            "side": (640, 480),
            "wrist": (640, 480),
        }
        for name in list(ctx.env.robot._renderers):
            ctx.env.robot._renderers[name].close()
            del ctx.env.robot._renderers[name]
        for name in ctx.env.task.camera_names:
            h, w = ctx.env.robot._camera_resolution(name)
            ctx.env.robot._renderers[name] = mj.Renderer(ctx.env.model, height=h, width=w)
        if ctx.env._depth_renderer is not None:
            ctx.env._depth_renderer.close()
            h, w = ctx.env.robot._camera_resolution("top")
            ctx.env._depth_renderer = mj.Renderer(ctx.env.model, height=h, width=w)
            ctx.env._depth_renderer.enable_depth_rendering()

        recorder = PlaceRecorder(
            ctx.env.robot,
            repo_id=repo_id,
            root=root,
            save_failures=save_failures,
            depth_env=ctx.env,
            record_top_depth=record_top_depth,
        )
        print(
            f"record_top_depth={recorder.record_top_depth} "
            f"features={sorted(k for k in recorder.features if 'images' in k)}",
            flush=True,
        )
        _install_step_hook(ctx.env, recorder)

        attempts = 0
        while recorder.saved + already < target and attempts < max_attempts:
            attempts += 1
            scene_i = (ctx._index) % max(len(ctx.scenes), 1)
            layout = ctx.scenes[scene_i] if ctx.scenes else {}
            recorder.begin(
                {
                    "scene_index": scene_i,
                    "x_m": layout.get("x_m"),
                    "y_m": layout.get("y_m"),
                    "yaw_deg": layout.get("yaw_deg"),
                    "attempt": attempts,
                }
            )
            try:
                ok = bool(place_main(ctx))
            except Exception as exc:  # noqa: BLE001
                print(f"attempt {attempts} error: {type(exc).__name__}: {exc}", flush=True)
                ok = False
            # Episode index inside env after reset inside place_main.
            ep = int(ctx.env.episode)
            recorder.end(success=ok, episode=ep)
            print(
                f"attempt {attempts}: success={ok} saved={recorder.saved + already}/{target}",
                flush=True,
            )

        recorder.close()
        total = recorder.saved + already
        return {
            "saved": total,
            "attempts": attempts,
            "discarded": recorder.skipped,
            "done": total >= target,
            "root": str(root),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/so101-simstudio-rax-drawer-place100d"),
    )
    parser.add_argument(
        "--repo-id",
        default="alexhegit/so101-simstudio-rax-drawer-place100d",
    )
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=400)
    parser.add_argument(
        "--no-top-depth",
        action="store_true",
        help="Do not record camera_top_depth (RGB-only dataset).",
    )
    args = parser.parse_args()
    lock_path = Path(args.root).with_suffix(".collect.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_path, "w")
    try:
        import fcntl

        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise SystemExit(f"another collector holds {lock_path}: {exc}") from exc
    lock_fd.write(f"pid={os.getpid()}\n")
    lock_fd.flush()
    try:
        info = collect(
            target=args.target,
            root=args.root,
            repo_id=args.repo_id,
            save_failures=args.save_failures,
            max_attempts=args.max_attempts,
            record_top_depth=not args.no_top_depth,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
    print(json.dumps(info, indent=2))
    if not info.get("done"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
