#!/usr/bin/env python3
"""Parse ACT training loss from lerobot train log and save CSV + PNG."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_step_token(raw: str) -> int:
    raw = raw.strip()
    if raw.endswith("K"):
        return int(float(raw[:-1]) * 1000)
    if raw.endswith("M"):
        return int(float(raw[:-1]) * 1_000_000)
    return int(raw)


def parse_loss_log(log_path: Path, total_steps: int) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    seen: set[int] = set()
    pattern = re.compile(
        r"ot_train\.py:606.*step:([\d.]+[KM]?) .*loss:([\d.]+)",
    )
    for line in log_path.read_text(errors="replace").splitlines():
        if "ot_train.py:606" not in line:
            continue
        match = pattern.search(line)
        if not match:
            continue
        step = _parse_step_token(match.group(1))
        loss = float(match.group(2))
        if step > total_steps or step in seen:
            continue
        seen.add(step)
        steps.append(step)
        losses.append(loss)
    return steps, losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("train_act.log"),
        help="Training log path (default: train_act.log)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Total training steps (default: 50000)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("labs/lab01_pnp"),
        help="Output directory for CSV and PNG",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="mi300x_50k",
        help="Filename tag, e.g. mi300x_50k -> loss_curve_act_mi300x_50k.png",
    )
    args = parser.parse_args()

    steps, losses = parse_loss_log(args.log, args.steps)
    if not steps:
        raise SystemExit(f"No loss entries found in {args.log}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"loss_curve_act_{args.tag}.csv"
    png_path = args.out_dir / f"loss_curve_act_{args.tag}.png"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("step,loss\n")
        for step, loss in zip(steps, losses, strict=True):
            f.write(f"{step},{loss}\n")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, linewidth=1.2, color="#2563eb")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"ACT training loss ({args.tag}, n={len(steps)})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, args.steps)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"Parsed {len(steps)} points: step {steps[0]}..{steps[-1]}")
    print(f"First loss: {losses[0]:.3f}, last loss: {losses[-1]:.3f}")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
