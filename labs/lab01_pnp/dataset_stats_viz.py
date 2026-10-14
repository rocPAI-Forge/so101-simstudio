#!/usr/bin/env python3
"""Compare cube-spawn coverage and wrist usage across the Lab 01 datasets.

Renders two PNGs into ``labs/lab01_pnp_agent/viz/``:

* ``coverage_topview.png`` — workspace top view, per-episode cube pose and yaw.
* ``coverage_stats.png``   — wrist-roll range, yaw distribution, episode length.

Robot geometry (base pivot, home end-effector) is read from the MJCF so the
orientation marker reflects the real scene instead of a hand-typed guess.

Cube poses come from ``estimate_cube_pose.py``, which recovers them from the
top camera because only the yaw-aware collector logged them. Run that first::

    .venv-rocm/bin/python labs/lab01_pnp/estimate_cube_pose.py --report
    .venv-rocm/bin/python labs/lab01_pnp/dataset_stats_viz.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import pyarrow.parquet as pq
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "datasets"
SCENE = REPO / "SO101" / "scenes" / "simple_pick" / "scene.xml"
OUT_DIR = REPO / "labs" / "lab01_pnp_agent" / "viz"
POSE_JSON = OUT_DIR / "cube_pose_estimates.json"

TELEOP = "so101-simstudio-lab01-pnp"
AGENT = "so101-simstudio-lab01-pnp-agent"
AGENT_YAW = "so101-simstudio-lab01-pnp-agent-yaw"

# Randomization bounds used by both scripted collection campaigns.
CUBE_X = (0.26, 0.34)
CUBE_Y = (0.165, 0.235)
YAW_EDGES = np.arange(-45.0, 46.0, 15.0)

HOME_JOINTS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.3,
    "elbow_flex": 0.6,
    "wrist_flex": 1.2,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

C_TELEOP = "#7B64B8"
C_AGENT = "#2E79B5"
C_YAW = "#1F8A65"


@dataclass
class ActionStats:
    """Wrist-roll and episode-length summary derived from action vectors."""

    episodes: int
    frames: int
    close_roll: np.ndarray
    lengths: np.ndarray


def home_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Return the base pivot XY and the home end-effector XY in world frame."""
    model = mj.MjModel.from_xml_path(str(SCENE))
    data = mj.MjData(model)
    for name, value in HOME_JOINTS.items():
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = value
    mj.mj_forward(model, data)

    pan_jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder_pan")
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "gripperframe")
    return data.xanchor[pan_jid][:2].copy(), data.site_xpos[site_id][:2].copy()


def action_stats(name: str) -> ActionStats:
    """Per-episode wrist roll at the first gripper-close transition."""
    files = sorted((DATASETS / name / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {DATASETS / name / 'data'}")

    episodes: list[np.ndarray] = []
    for path in files:
        table = pq.read_table(path, columns=["episode_index", "action"])
        ep = table.column("episode_index").to_numpy()
        act = np.asarray(table.column("action").to_pylist(), dtype=float)
        for value in np.unique(ep):
            episodes.append(act[ep == value])

    rolls, lengths = [], []
    for act in episodes:
        grip = act[:, 5]
        closing = np.where((grip[1:] < 0.3) & (grip[:-1] >= 0.3))[0] + 1
        idx = int(closing[0]) if closing.size else int(np.argmin(grip))
        rolls.append(float(act[idx, 4]))
        lengths.append(len(act))

    return ActionStats(
        episodes=len(episodes),
        frames=int(sum(lengths)),
        close_roll=np.asarray(rolls),
        lengths=np.asarray(lengths),
    )


def load_poses() -> dict:
    """Per-episode cube poses recovered by estimate_cube_pose.py."""
    if not POSE_JSON.exists():
        raise FileNotFoundError(f"{POSE_JSON} missing — run labs/lab01_pnp/estimate_cube_pose.py first")
    return json.loads(POSE_JSON.read_text())


def _frame_axes(ax, base_xy: np.ndarray) -> None:
    ax.add_patch(
        Rectangle(
            (CUBE_X[0], CUBE_Y[0]),
            CUBE_X[1] - CUBE_X[0],
            CUBE_Y[1] - CUBE_Y[0],
            fill=False,
            edgecolor="#555",
            linestyle="--",
            linewidth=1.0,
        )
    )
    ax.plot(*base_xy, marker="x", color="#C06028", markersize=7, zorder=5)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.18)


def _draw_cubes(ax, poses: list[dict], cmap, norm) -> None:
    """Rotated cube footprints; a short line marks the graspable face axis."""
    half = 0.0045  # cube is 30 mm wide; drawn at ~1/3 scale so markers stay apart
    for pose in poses:
        rad = np.radians(pose["yaw_deg"])
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]]) @ rot.T
        corners += (pose["x"], pose["y"])
        color = cmap(norm(pose["yaw_deg"]))
        ax.fill(corners[:, 0], corners[:, 1], color=color, alpha=0.85, zorder=3)
        axis = rot @ np.array([half * 2.1, 0.0])
        ax.plot(
            [pose["x"] - axis[0], pose["x"] + axis[0]],
            [pose["y"] - axis[1], pose["y"] + axis[1]],
            color=color,
            lw=1.0,
            zorder=4,
        )


def plot_topview(base_xy: np.ndarray, home_ee: np.ndarray, poses: dict) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(21.5, 5.4))
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(-45, 45)

    # Panel 1: scene geometry with the arm orientation triangle.
    ax = axes[0]
    forward = home_ee - base_xy
    forward /= np.linalg.norm(forward)
    side = np.array([-forward[1], forward[0]])
    apex = base_xy + forward * 0.085
    tri = np.array([apex, base_xy - side * 0.038, base_xy + side * 0.038])
    ax.fill(tri[:, 0], tri[:, 1], color="#C06028", alpha=0.55, zorder=3)
    ax.plot(tri[[0, 1, 2, 0], 0], tri[[0, 1, 2, 0], 1], color="#C06028", lw=1.4, zorder=4)
    ax.annotate(
        "apex = home forward axis",
        xy=apex,
        xytext=(apex[0] + 0.055, apex[1] - 0.045),
        arrowprops={"arrowstyle": "->", "color": "#C06028", "lw": 1.1},
        fontsize=9,
        color="#C06028",
    )
    ax.plot(*home_ee, marker="o", ms=6, color="#C06028", mfc="white", zorder=5)
    ax.annotate(
        "home gripper",
        xy=home_ee,
        xytext=(home_ee[0] + 0.055, home_ee[1] + 0.045),
        arrowprops={"arrowstyle": "->", "color": "#666", "lw": 0.9},
        fontsize=9,
    )
    ax.add_patch(
        Rectangle((0.07 - 0.045, 0.065 - 0.045), 0.09, 0.09, fill=False, edgecolor="#8a6b45", lw=1.4)
    )
    ax.annotate("container", xy=(0.07, 0.065), ha="center", va="center", fontsize=9, color="#8a6b45")
    _frame_axes(ax, base_xy)
    ax.annotate(
        "base pivot", xy=base_xy, xytext=(base_xy[0] + 0.02, base_xy[1] - 0.035), fontsize=9, color="#C06028"
    )
    ax.annotate("cube envelope", xy=(0.30, CUBE_Y[1]), xytext=(0.245, 0.275), fontsize=9, color="#555")
    ax.set_xlim(-0.03, 0.42)
    ax.set_ylim(-0.17, 0.30)
    ax.set_title("Scene geometry (home pose)\ntriangle apex = arm initial direction", fontsize=11)

    # Panels 2-4: recovered cube pose per episode, one panel per dataset.
    panels = [
        ("teleop", "teleop (leader)", C_TELEOP),
        ("agent", "agent (fixed wrist)", C_AGENT),
        ("yaw", "agent-yaw", C_YAW),
    ]
    for ax, (key, label, color) in zip(axes[1:], panels, strict=True):
        data = poses["datasets"][key]
        yaws = np.array([p["yaw_deg"] for p in data])
        _draw_cubes(ax, data, cmap, norm)
        _frame_axes(ax, base_xy)
        ax.set_xlim(0.252, 0.348)
        ax.set_ylim(0.158, 0.242)
        ax.set_title(
            f"{label} · {len(data)} episodes\n"
            f"yaw {yaws.min():+.0f}..{yaws.max():+.0f}°, σ = {yaws.std():.1f}°",
            fontsize=11,
            color=color,
        )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes[-1],
        label="cube reset yaw (deg)",
        fraction=0.046,
    )

    cal = poses["calibration"]
    fig.suptitle(
        "Lab 01 pick-and-place · cube reset pose per episode, recovered from the top camera "
        f"(yaw error {cal['yaw_err_deg_mean']:+.2f} ± {cal['yaw_err_deg_std']:.2f}°, "
        f"position σ ≈ {max(cal['residual_x_mm_std'], cal['residual_y_mm_std']):.1f} mm "
        "vs agent-yaw logged truth)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / "coverage_topview.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_stats(stats: dict[str, ActionStats], poses: dict) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(21.5, 4.6))
    labels = ["teleop", "agent\n(fixed wrist)", "agent-yaw"]
    keys = ["teleop", "agent", "yaw"]
    colors = [C_TELEOP, C_AGENT, C_YAW]

    # Wrist roll at first close — the discriminating variable for grasp alignment.
    ax = axes[0]
    for i, (key, color) in enumerate(zip(keys, colors, strict=True)):
        roll = stats[key].close_roll
        ax.plot([roll.min(), roll.max()], [i, i], color=color, lw=7, solid_capstyle="round", alpha=0.75)
        ax.plot(
            roll,
            np.full_like(roll, i) + np.random.default_rng(0).normal(0, 0.055, roll.size),
            ".",
            color=color,
            ms=4,
            alpha=0.65,
        )
        ax.plot(roll.mean(), i, "o", color="white", mec=color, mew=2, ms=9, zorder=5)
        ax.annotate(
            f"σ = {roll.std():.3f} rad",
            xy=(roll.max(), i),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )
    ax.set_yticks(range(3), labels)
    ax.invert_yaxis()
    ax.set_xlabel("commanded wrist_roll at first gripper close (rad)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("Wrist-roll usage per episode\nwhite dot = mean, bar = min..max", fontsize=11)

    # Yaw coverage as a share of each dataset, so 50 and 100 episodes compare.
    ax = axes[1]
    centers = YAW_EDGES[:-1] + 7.5
    width = 4.0
    for offset, (key, color, label) in enumerate(
        zip(keys, colors, ["teleop", "agent (fixed wrist)", "agent-yaw"], strict=True)
    ):
        yaws = np.array([p["yaw_deg"] for p in poses["datasets"][key]])
        share = np.histogram(yaws, bins=YAW_EDGES)[0] / len(yaws) * 100
        ax.bar(centers + (offset - 1) * width, share, width=width, color=color, alpha=0.85, label=label)
    ax.axhline(100 / 6, color="#333", ls=":", lw=1.2, label="uniform coverage (16.7%)")
    ax.set_xticks(centers, [f"{int(e)}..{int(e) + 15}" for e in YAW_EDGES[:-1]])
    ax.set_xlabel("cube reset yaw bin (deg)")
    ax.set_ylabel("share of dataset episodes (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title(
        "Cube yaw coverage\nagent has zero episodes above +30°",
        fontsize=11,
    )

    # The coupling the policy has to learn: wrist orientation as a function of
    # cube yaw. A flat cloud means the demonstrations never taught it.
    ax = axes[2]
    for key, color, label in zip(keys, colors, ["teleop", "agent (fixed wrist)", "agent-yaw"], strict=True):
        yaws = np.array([p["yaw_deg"] for p in poses["datasets"][key]])
        roll = stats[key].close_roll
        corr = float(np.corrcoef(yaws, roll)[0, 1]) if roll.std() > 1e-9 else 0.0
        ax.scatter(
            yaws, roll, s=22, color=color, alpha=0.7, edgecolor="none", label=f"{label} (r = {corr:+.2f})"
        )
    ax.axhline(0.17, color=C_AGENT, ls=":", lw=1.2)
    ax.set_xlabel("cube reset yaw (deg)")
    ax.set_ylabel("wrist_roll at first close (rad)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    # The yaw-aware solver picks the nearest of the cube's 90-deg-equivalent
    # faces, so its high-yaw episodes branch to a negative roll instead of
    # continuing up. That branch is why its r is well below 1.
    ax.annotate(
        "green branch = ±90° face\nequivalent, not a failure",
        xy=(38, -0.5),
        xytext=(6, 28),
        textcoords="offset points",
        fontsize=8,
        color=C_YAW,
        ha="right",
    )
    ax.set_title("Does the wrist follow the cube?\neach dot is one episode", fontsize=11)

    # Episode length distribution.
    ax = axes[3]
    parts = ax.violinplot([stats[k].lengths for k in keys], showmeans=False, showextrema=False)
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    for i, key in enumerate(keys, start=1):
        med = float(np.median(stats[key].lengths))
        ax.plot(i, med, "o", color="white", mec="#333", mew=1.4, ms=8, zorder=5)
        ax.annotate(f"{med:.0f}", xy=(i, med), xytext=(9, -4), textcoords="offset points", fontsize=9)
    ax.set_xticks(range(1, 4), labels)
    ax.set_ylabel("frames per episode (20 fps)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("Episode length distribution\nwhite dot = median", fontsize=11)

    fig.suptitle("Lab 01 pick-and-place · action-space statistics across three datasets", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT_DIR / "coverage_stats.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_xy, home_ee = home_geometry()
    stats = {"teleop": action_stats(TELEOP), "agent": action_stats(AGENT), "yaw": action_stats(AGENT_YAW)}
    poses = load_poses()

    p1 = plot_topview(base_xy, home_ee, poses)
    p2 = plot_stats(stats, poses)

    print(f"base pivot XY   : ({base_xy[0]:.4f}, {base_xy[1]:.4f})")
    print(f"home gripper XY : ({home_ee[0]:.4f}, {home_ee[1]:.4f})")
    for key, label in (("teleop", TELEOP), ("agent", AGENT), ("yaw", AGENT_YAW)):
        s = stats[key]
        yaws = np.array([p["yaw_deg"] for p in poses["datasets"][key]])
        print(
            f"{label:42s} ep={s.episodes:3d} frames={s.frames:6d} "
            f"roll={s.close_roll.mean():+.3f}±{s.close_roll.std():.3f} rad  "
            f"yaw {yaws.mean():+.1f}±{yaws.std():.1f}° "
            f"[{yaws.min():+.1f}, {yaws.max():+.1f}]"
        )
    cal = poses["calibration"]
    print(
        f"pose recovery vs logged truth: yaw {cal['yaw_err_deg_mean']:+.2f}±"
        f"{cal['yaw_err_deg_std']:.2f}°, x σ {cal['residual_x_mm_std']:.2f} mm, "
        f"y σ {cal['residual_y_mm_std']:.2f} mm (n={cal['n']})"
    )
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
