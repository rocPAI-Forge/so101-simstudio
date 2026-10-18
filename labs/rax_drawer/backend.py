"""Backend compatible with RAX real/franka robot-service tools."""

from __future__ import annotations

import base64
import math
import struct
import zlib
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from .environment import DrawerTaskConfig, RaxDrawerEnv


def _png(rgb: np.ndarray) -> bytes:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got {image.shape}")
    height, width, _ = image.shape

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + image[y].tobytes() for y in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _pack(value: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    value = dict(value)
    images = []
    cameras = value.pop("cameras", None)
    nested = value.get("observation")
    if cameras is None and isinstance(nested, dict):
        nested = dict(nested)
        cameras = nested.pop("cameras", None)
        if cameras is not None:
            nested["camera_names"] = list(cameras)
            value["observation"] = nested
    elif cameras is not None:
        value["camera_names"] = list(cameras)
    if cameras:
        images = [
            {"type": "image", "mimeType": "image/png", "data": base64.b64encode(_png(frame)).decode("ascii")}
            for frame in cameras.values()
        ]
        value["camera_image_count"] = len(images)
    return value, images


class RaxDrawerBackend:
    """RAX robot-service backend exposing drawer tools."""

    mutating = frozenset({"drawer_reset", "drawer_step"})

    def __init__(self, config: Any = None):
        if isinstance(config, DrawerTaskConfig):
            task = config
        else:
            options = config if isinstance(config, dict) else getattr(config, "backend_options", {})
            if not isinstance(options, dict):
                raise ValueError("backend_options must be an object")
            allowed = {field.name for field in fields(DrawerTaskConfig)}
            unknown = set(options) - allowed
            if unknown:
                raise ValueError(f"unknown drawer backend options: {sorted(unknown)}")
            values = dict(options)
            if "scene_path" in values:
                values["scene_path"] = Path(values["scene_path"])
            if "camera_names" in values:
                if isinstance(values["camera_names"], str):
                    raise ValueError("camera_names must be a list")
                values["camera_names"] = tuple(values["camera_names"])
            task = DrawerTaskConfig(**values)
        self.env = RaxDrawerEnv(task)

    def tools(self) -> list[dict[str, Any]]:
        descriptions = {
            "vx": "World X velocity in m/s.",
            "vy": "World Y velocity in m/s; positive Y pulls toward the robot and opens the drawer.",
            "vz": "World Z velocity in m/s.",
            "wrist_flex_rate": "Wrist flex rate in rad/s.",
            "yaw_rate": "Wrist roll rate in rad/s.",
            "gripper_delta": "Gripper rate; negative closes, positive opens.",
        }
        actions = {k: {"type": "number", "description": descriptions[k]} for k in self.env.ACTION_LIMITS}
        return [
            {
                "name": "drawer_observe",
                "description": "Observe joint state, end-effector pose, drawer travel, and camera frames.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"include_cameras": {"type": "boolean", "default": True}},
                },
            },
            {
                "name": "drawer_reset",
                "description": "Reset the robot and initialize a fresh, deterministic drawer episode.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "seed": {"type": "integer"},
                        "drawer_position_m": {
                            "type": "number",
                            "minimum": self.env.bounds[0],
                            "maximum": self.env.bounds[1] - self.env.task.success_position_m,
                            "default": 0,
                        },
                        "include_cameras": {"type": "boolean", "default": True},
                    },
                },
            },
            {
                "name": "drawer_step",
                "description": (
                    "Apply bounded velocity commands. Close on the handle, then use "
                    f"positive world Y to pull open. Observe between actions; max duration "
                    f"{self.env.task.max_action_duration_s:g} s."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "object", "properties": actions, "additionalProperties": False},
                        "duration_s": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": self.env.task.max_action_duration_s,
                            "default": 0.25,
                        },
                        "include_cameras": {"type": "boolean", "default": True},
                    },
                    "required": ["action"],
                },
            },
            {
                "name": "drawer_check_success",
                "description": (
                    f"Check for at least {self.env.task.success_position_m * 100:.0f} cm "
                    "of drawer travel since reset."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

    def validate(self, name: str, args: dict[str, Any]) -> None:
        allowed = {
            "drawer_observe": {"include_cameras"},
            "drawer_reset": {"seed", "drawer_position_m", "include_cameras"},
            "drawer_step": {"action", "duration_s", "include_cameras"},
            "drawer_check_success": set(),
        }
        if name not in allowed:
            raise ValueError(f"unknown drawer tool: {name}")
        if set(args) - allowed[name]:
            raise ValueError(f"unexpected arguments for {name}")
        if "include_cameras" in args and type(args["include_cameras"]) is not bool:
            raise ValueError("include_cameras must be a boolean")
        if name == "drawer_reset":
            seed = args.get("seed")
            if seed is not None and type(seed) is not int:
                raise ValueError("seed must be an integer")
            position = float(args.get("drawer_position_m", 0.0))
            max_start = self.env.bounds[1] - self.env.task.success_position_m
            if not math.isfinite(position) or not self.env.bounds[0] <= position <= max_start:
                raise ValueError("drawer start leaves too little travel to reach success")
        elif name == "drawer_step":
            action = args.get("action")
            if not isinstance(action, dict):
                raise ValueError("action must be an object")
            duration = float(args.get("duration_s", 0.25))
            if not math.isfinite(duration) or not 0 < duration <= self.env.task.max_action_duration_s:
                raise ValueError("duration_s is outside the allowed range")
            self.env._bound_action(action)

    def call(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        self.validate(name, args)
        if name == "drawer_observe":
            value = self.env.observe(include_cameras=args.get("include_cameras", True))
        elif name == "drawer_reset":
            value = self.env.reset(
                seed=args.get("seed"),
                drawer_position_m=args.get("drawer_position_m", 0.0),
                include_cameras=args.get("include_cameras", True),
            )
        elif name == "drawer_step":
            value = self.env.step(
                args["action"],
                duration_s=args.get("duration_s", 0.25),
                include_cameras=args.get("include_cameras", True),
            )
        else:
            position = self.env.drawer_position()
            value = {
                "success": self.env.check_success(),
                "drawer_position_m": position,
                "drawer_opened_distance_m": position - self.env.start_drawer_position,
                "success_threshold_m": self.env.task.success_position_m,
            }
        return _pack(value)

    def status(self) -> dict[str, Any]:
        position = self.env.drawer_position()
        return {
            "robot": "so101_mujoco",
            "task": "pull_open_smart_drawer",
            "episode": self.env.episode,
            "drawer_position_m": position,
            "drawer_opened_distance_m": position - self.env.start_drawer_position,
            "success_threshold_m": self.env.task.success_position_m,
            "success": self.env.check_success(),
        }

    def close(self) -> None:
        self.env.close()
