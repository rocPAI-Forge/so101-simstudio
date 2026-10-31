"""Render a full open → place → close GIF with top RGB, side RGB, and top depth.

Uses the frozen tool-only controller. Default scene is locked test index 1
(first success in the 15/20 freeze).

  MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.rax_place.render_place_gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from labs.rax_drawer.rax_place.code.open_place_close.controller import main
from labs.rax_drawer.rax_place.tools import Session

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE.parent / "media" / "open_place_close_rgbd.gif"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _depth_rgb(depth_m: np.ndarray) -> np.ndarray:
    """Map metric depth to a readable gray image for the GIF."""
    valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 2.0)
    out = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    lo = float(np.percentile(depth_m[valid], 5))
    hi = float(np.percentile(depth_m[valid], 95))
    hi = max(hi, lo + 1e-3)
    norm = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
    gray = (255.0 * (1.0 - norm)).astype(np.uint8)
    gray[~valid] = 0
    out[..., 0] = gray
    out[..., 1] = gray
    out[..., 2] = gray
    return out


def _resize(rgb: np.ndarray, width: int, height: int) -> Image.Image:
    return Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR)


def render(*, scene_index: int, split: str, out: Path, interval_s: float) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 16)
    small = ImageFont.truetype(FONT, 13)
    frames: list[Image.Image] = []
    last_t = -1e9
    panel_w, top_h, side_h, bar = 320, 240, 240, 28

    with Session(
        split=split,
        episode_limit_s=180.0,
        audit_path=HERE / "logs" / "mcp_render_gif.jsonl",
    ) as ctx:
        # Next reset inside main() loads this scene index.
        ctx._index = int(scene_index)
        env = ctx.env
        side_r = mj.Renderer(env.model, height=side_h, width=panel_w * 2)
        top_r = mj.Renderer(env.model, height=top_h, width=panel_w)
        depth_r = mj.Renderer(env.model, height=top_h, width=panel_w)
        depth_r.enable_depth_rendering()

        def grab(force: bool = False) -> None:
            nonlocal last_t
            if env.episode < 0:
                return
            t = float(env.data.time - env.start_time)
            if frames and not force and t - last_t < interval_s:
                return
            last_t = t
            top_r.update_scene(env.data, camera="top")
            side_r.update_scene(env.data, camera="side")
            depth_r.update_scene(env.data, camera="top")
            top = _resize(np.asarray(top_r.render(), dtype=np.uint8), panel_w, top_h)
            side = _resize(np.asarray(side_r.render(), dtype=np.uint8), panel_w * 2, side_h)
            depth = _resize(_depth_rgb(np.asarray(depth_r.render(), dtype=np.float32)), panel_w, top_h)
            canvas = Image.new("RGB", (panel_w * 2, bar + top_h + side_h), (24, 26, 28))
            canvas.paste(top, (0, bar))
            canvas.paste(depth, (panel_w, bar))
            canvas.paste(side, (0, bar + top_h))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (8, 5),
                f"{split} {scene_index}   t={t:5.1f}s   open → place → close",
                fill=(240, 240, 240),
                font=font,
            )
            draw.text((8, bar + 4), "top RGB", fill=(220, 220, 220), font=small)
            draw.text((panel_w + 8, bar + 4), "top depth", fill=(220, 220, 220), font=small)
            draw.text((8, bar + top_h + 4), "side RGB", fill=(220, 220, 220), font=small)
            frames.append(canvas)

        orig_step = ctx.step
        orig_reset = ctx.reset

        def stepped(*args, **kwargs):
            result = orig_step(*args, **kwargs)
            grab()
            return result

        def reset_hook(*args, **kwargs):
            result = orig_reset(*args, **kwargs)
            grab(force=True)
            return result

        ctx.step = stepped  # type: ignore[method-assign]
        ctx.reset = reset_hook  # type: ignore[method-assign]
        ok = bool(main(ctx))
        grab(force=True)
        stamp = "success" if ok else "fail"
        color = (120, 220, 140) if ok else (240, 120, 120)
        for frame in frames[-6:]:
            draw = ImageDraw.Draw(frame)
            draw.text((panel_w * 2 - 90, 5), stamp, fill=color, font=font)

        frames[0].save(
            out,
            save_all=True,
            append_images=frames[1:],
            duration=110,
            loop=0,
            optimize=True,
        )
        side_r.close()
        top_r.close()
        depth_r.close()
        return {
            "success": ok,
            "frames": len(frames),
            "bytes": out.stat().st_size,
            "path": str(out),
            "cube_in_drawer": bool(env.cube_in_drawer()),
        }


def main_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("explore", "test"))
    parser.add_argument("--scene", type=int, default=1, help="scene index in the split")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--interval-s", type=float, default=0.40)
    args = parser.parse_args()
    info = render(scene_index=args.scene, split=args.split, out=args.out, interval_s=args.interval_s)
    print(info)


if __name__ == "__main__":
    main_cli()
