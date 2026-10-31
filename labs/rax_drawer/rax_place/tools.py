"""Non-privileged RAX session tools for the open-place-close task.

The agent may call only reset / observe / step / check_success. Cabinet layout
comes from the explore or test manifest behind the scenes and is never returned.
Drawer joint, cube pose, handle pose, and contacts stay inside the evaluator.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from labs.rax_drawer.backend import RaxDrawerBackend
from labs.rax_drawer.environment import DrawerTaskConfig

ROOT = Path(__file__).resolve().parents[1]
SESSION = Path(__file__).resolve().parent
MANIFESTS = ROOT / "manifests"
AUDIT = SESSION / "logs" / "mcp.jsonl"


def _strip_for_agent(value: dict[str, Any]) -> dict[str, Any]:
    """Keep the public observation contract; drop evaluator-only fields if any sneak in."""
    banned = {
        "drawer_position_m",
        "drawer_opened_distance_m",
        "cube_position_m",
        "handle_position_m",
        "layout",
        "pull_contact",
        "close_contact",
        "reached_open",
        "cube_placed",
        "cube_in_drawer",
    }
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key in banned:
            continue
        if isinstance(item, dict):
            out[key] = _strip_for_agent(item)
        else:
            out[key] = item
    return out


class Session:
    """One robot-owned session. Layouts are applied privately on reset."""

    def __init__(
        self,
        *,
        split: str = "explore",
        episode_limit_s: float = 180.0,
        audit_path: Path | None = None,
    ):
        if split not in {"explore", "test", "identity"}:
            raise ValueError("split must be explore, test, or identity")
        self.split = split
        self.audit_path = Path(audit_path) if audit_path else AUDIT
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.scenes: list[dict[str, Any]] = []
        if split in {"explore", "test"}:
            with open(MANIFESTS / f"{split}.json") as handle:
                self.scenes = json.load(handle)["scenes"]
        self._index = 0
        self._episode = -1
        self._private_layout: dict[str, float] | None = None
        task = DrawerTaskConfig(
            require_cube_placed=True,
            episode_limit_s=float(episode_limit_s),
            camera_names=("top", "side", "wrist"),
        )
        self.backend = RaxDrawerBackend(task)
        self.env = self.backend.env
        self._log(
            "session_start",
            {
                "split": split,
                "episode_limit_s": episode_limit_s,
                "n_scenes": len(self.scenes),
                "task": asdict(task),
            },
        )

    def _log(self, name: str, payload: dict[str, Any]) -> None:
        row = {
            "ts": time.time(),
            "split": self.split,
            "episode": self._episode,
            "tool": name,
            "payload": payload,
        }
        with open(self.audit_path, "a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    def _apply_next_layout(self) -> None:
        if not self.scenes:
            self._private_layout = {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0}
            self.env.set_layout(**self._private_layout)
            return
        scene = self.scenes[self._index % len(self.scenes)]
        self._index += 1
        self._private_layout = {
            "x_m": float(scene["x_m"]),
            "y_m": float(scene["y_m"]),
            "yaw_deg": float(scene["yaw_deg"]),
        }
        self.env.set_layout(**self._private_layout)

    def reset(self, *, drawer_position_m: float = 0.0, include_cameras: bool = False) -> dict[str, Any]:
        self._apply_next_layout()
        self.backend.validate(
            "drawer_reset",
            {
                "seed": None,
                "drawer_position_m": float(drawer_position_m),
                "include_cameras": bool(include_cameras),
            },
        )
        value = self.env.reset(
            drawer_position_m=float(drawer_position_m),
            include_cameras=bool(include_cameras),
        )
        self._episode = int(self.env.episode)
        public = _strip_for_agent(value)
        self._log(
            "drawer_reset",
            {
                "args": {"drawer_position_m": drawer_position_m, "include_cameras": include_cameras},
                "private_layout_index": (self._index - 1) if self.scenes else None,
                "result_keys": sorted(public),
            },
        )
        return public

    def observe(self, *, include_cameras: bool = True) -> dict[str, Any]:
        self.backend.validate("drawer_observe", {"include_cameras": bool(include_cameras)})
        value = self.env.observe(include_cameras=bool(include_cameras))
        public = _strip_for_agent(value)
        self._log(
            "drawer_observe",
            {
                "args": {"include_cameras": include_cameras},
                "has_cameras": "cameras" in public,
                "has_depth": "depth_images" in public,
            },
        )
        return public

    def step(
        self,
        action: dict[str, float],
        *,
        duration_s: float = 0.25,
        include_cameras: bool = False,
    ) -> dict[str, Any]:
        args = {
            "action": dict(action),
            "duration_s": float(duration_s),
            "include_cameras": bool(include_cameras),
        }
        self.backend.validate("drawer_step", args)
        value = self.env.step(
            args["action"],
            duration_s=args["duration_s"],
            include_cameras=args["include_cameras"],
        )
        public = _strip_for_agent(value)
        # Nested observation may still carry cameras; keep them for the agent.
        self._log(
            "drawer_step",
            {
                "args": {
                    "action": action,
                    "duration_s": duration_s,
                    "include_cameras": include_cameras,
                },
                "terminated": public.get("terminated"),
                "success": public.get("success"),
            },
        )
        return public

    def check_success(self) -> dict[str, Any]:
        self.backend.validate("drawer_check_success", {})
        public = {"success": bool(self.env.check_success())}
        self._log("drawer_check_success", {"result": public})
        return public

    def close(self) -> None:
        self._log("session_close", {})
        self.backend.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
