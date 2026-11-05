"""Render ACT RGB vs RGB-D side-by-side camera collage GIF.

Layout (column-aligned 2x3; wrist omitted — flipped/awkward for this montage):

  RGB   | top | side |  — (no depth)
  RGB-D | top | side | top_depth

Default scene: locked test index 6 (both policies succeeded in place100d eval).

  MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.vla.render_policy_compare_gif
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco as mj
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from labs.rax_drawer.environment import DrawerTaskConfig, RaxDrawerEnv
from labs.rax_drawer.vla.depth_vis import depth_m_to_rgb3
from labs.rax_drawer.vla.eval_policy import JOINTS, _load_policy, _obs_batch

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
OUT_DEFAULT = LAB / "media" / "act_rgb_vs_rgbd_compare.gif"

RGB_CKPT = Path("outputs/train/rax_drawer_act_rgb/checkpoints/030000/pretrained_model")
RGBD_CKPT = Path("outputs/train/rax_drawer_act_rgbd/checkpoints/030000/pretrained_model")


def _resize(rgb: np.ndarray, w: int, h: int) -> Image.Image:
    return Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize((w, h), Image.Resampling.BILINEAR)


def _blank(w: int, h: int, text: str, font: ImageFont.ImageFont) -> Image.Image:
    img = Image.new("RGB", (w, h), (36, 38, 42))
    draw = ImageDraw.Draw(img)
    draw.text((12, h // 2 - 8), text, fill=(160, 160, 160), font=font)
    return img


def _make_env(max_s: float) -> RaxDrawerEnv:
    task = DrawerTaskConfig(
        require_cube_placed=True,
        episode_limit_s=max_s + 5.0,
        render_window=False,
    )
    env = RaxDrawerEnv(task)
    env.robot.config.camera_resolutions = {
        "top": (640, 480),
        "side": (640, 480),
        "wrist": (640, 480),
    }
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
    return env


def _run_and_capture(
    *,
    policy_path: Path,
    scene: dict,
    device: str,
    state_dim: int,
    max_s: float,
    interval_s: float,
    with_depth: bool,
) -> tuple[list[dict[str, np.ndarray]], bool]:
    policy, preprocess, postprocess, cfg = _load_policy(policy_path, device, state_dim=state_dim)
    image_keys = [
        key
        for key, ft in getattr(cfg, "input_features", {}).items()
        if key.startswith("observation.images.")
    ]
    env = _make_env(max_s)
    env.set_layout(x_m=scene["x_m"], y_m=scene["y_m"], yaw_deg=scene["yaw_deg"])
    env.reset(include_cameras=False)
    policy.reset()

    fps = float(env.robot.config.record_fps)
    dt = 1.0 / fps
    steps = int(max_s * fps)
    frames: list[dict[str, np.ndarray]] = []
    last_t = -1e9
    ok = False

    def grab(force: bool = False) -> None:
        nonlocal last_t
        t = float(env.data.time - env.start_time)
        if frames and not force and t - last_t < interval_s:
            return
        last_t = t
        raw = env.robot.get_observation()
        pack = {
            "top": np.asarray(raw["camera_top"], dtype=np.uint8),
            "side": np.asarray(raw["camera_side"], dtype=np.uint8),
            "t": np.array([t], dtype=np.float32),
        }
        if with_depth:
            pack["depth"] = depth_m_to_rgb3(env.top_depth_m())
        frames.append(pack)

    grab(force=True)
    for _ in range(steps):
        batch = _obs_batch(
            env, state_dim=state_dim, device=device, rename={}, image_keys=image_keys
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
        grab()
        if result.get("terminated"):
            ok = bool(result.get("success"))
            break
    else:
        ok = bool(env.check_success())
    grab(force=True)
    env.close()
    return frames, ok


def _compose(
    rgb_frames: list[dict[str, np.ndarray]],
    rgbd_frames: list[dict[str, np.ndarray]],
    *,
    rgb_ok: bool,
    rgbd_ok: bool,
    split: str,
    scene_index: int,
    panel_w: int,
    panel_h: int,
) -> list[Image.Image]:
    font = ImageFont.truetype(FONT, 15)
    small = ImageFont.truetype(FONT, 12)
    bar = 26
    label_h = 22
    n = max(len(rgb_frames), len(rgbd_frames))
    out: list[Image.Image] = []
    cols = 3
    w = panel_w * cols
    h = bar + label_h + panel_h + label_h + panel_h

    def pad(frames: list[dict[str, np.ndarray]], i: int) -> dict[str, np.ndarray]:
        if not frames:
            raise RuntimeError("empty frame list")
        return frames[min(i, len(frames) - 1)]

    for i in range(n):
        a = pad(rgb_frames, i)
        b = pad(rgbd_frames, i)
        canvas = Image.new("RGB", (w, h), (24, 26, 28))
        draw = ImageDraw.Draw(canvas)
        t_a = float(a["t"][0])
        t_b = float(b["t"][0])
        draw.text(
            (8, 4),
            f"{split} scene {scene_index}   RGB t={t_a:4.1f}s  RGB-D t={t_b:4.1f}s   "
            f"RGB={'OK' if rgb_ok else 'FAIL'}  RGB-D={'OK' if rgbd_ok else 'FAIL'}   "
            f"scores RGB 7/20  RGB-D 11/20",
            fill=(235, 235, 235),
            font=font,
        )
        y0 = bar
        draw.rectangle((0, y0, w, y0 + label_h), fill=(40, 44, 52))
        draw.text((8, y0 + 3), "ACT RGB (place100d) — top + side", fill=(200, 210, 230), font=small)
        y1 = y0 + label_h
        canvas.paste(_resize(a["top"], panel_w, panel_h), (0, y1))
        canvas.paste(_resize(a["side"], panel_w, panel_h), (panel_w, y1))
        canvas.paste(_blank(panel_w, panel_h, "no depth", small), (panel_w * 2, y1))
        for j, name in enumerate(("top", "side", "—")):
            draw.text((panel_w * j + 6, y1 + 4), name, fill=(230, 230, 230), font=small)

        y2 = y1 + panel_h
        draw.rectangle((0, y2, w, y2 + label_h), fill=(40, 52, 44))
        draw.text((8, y2 + 3), "ACT RGB-D (place100d) — top + side + top_depth", fill=(200, 230, 210), font=small)
        y3 = y2 + label_h
        canvas.paste(_resize(b["top"], panel_w, panel_h), (0, y3))
        canvas.paste(_resize(b["side"], panel_w, panel_h), (panel_w, y3))
        canvas.paste(_resize(b["depth"], panel_w, panel_h), (panel_w * 2, y3))
        for j, name in enumerate(("top", "side", "top_depth")):
            draw.text((panel_w * j + 6, y3 + 4), name, fill=(230, 230, 230), font=small)
        out.append(canvas)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("explore", "test"))
    parser.add_argument("--scene", type=int, default=6, help="scene index in the split")
    parser.add_argument("--rgb-policy", type=Path, default=RGB_CKPT)
    parser.add_argument("--rgbd-policy", type=Path, default=RGBD_CKPT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--max-s", type=float, default=90.0)
    parser.add_argument("--interval-s", type=float, default=0.40)
    parser.add_argument("--panel-w", type=int, default=240)
    parser.add_argument("--panel-h", type=int, default=180)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration-ms", type=int, default=110)
    args = parser.parse_args()

    manifests = LAB / "manifests" / f"{args.split}.json"
    scenes = json.loads(manifests.read_text())["scenes"]
    if args.scene < 0 or args.scene >= len(scenes):
        raise SystemExit(f"scene {args.scene} out of range 0..{len(scenes)-1}")
    scene = scenes[args.scene]

    t0 = time.time()
    print(f"capture RGB {args.rgb_policy} scene={args.scene}", flush=True)
    rgb_frames, rgb_ok = _run_and_capture(
        policy_path=args.rgb_policy,
        scene=scene,
        device=args.device,
        state_dim=6,
        max_s=args.max_s,
        interval_s=args.interval_s,
        with_depth=False,
    )
    print(f"RGB frames={len(rgb_frames)} success={rgb_ok}", flush=True)

    print(f"capture RGB-D {args.rgbd_policy} scene={args.scene}", flush=True)
    rgbd_frames, rgbd_ok = _run_and_capture(
        policy_path=args.rgbd_policy,
        scene=scene,
        device=args.device,
        state_dim=6,
        max_s=args.max_s,
        interval_s=args.interval_s,
        with_depth=True,
    )
    print(f"RGB-D frames={len(rgbd_frames)} success={rgbd_ok}", flush=True)

    frames = _compose(
        rgb_frames,
        rgbd_frames,
        rgb_ok=rgb_ok,
        rgbd_ok=rgbd_ok,
        split=args.split,
        scene_index=args.scene,
        panel_w=args.panel_w,
        panel_h=args.panel_h,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    print(
        {
            "path": str(args.out),
            "frames": len(frames),
            "bytes": args.out.stat().st_size,
            "rgb_ok": rgb_ok,
            "rgbd_ok": rgbd_ok,
            "elapsed_s": round(time.time() - t0, 1),
        }
    )


if __name__ == "__main__":
    main()
