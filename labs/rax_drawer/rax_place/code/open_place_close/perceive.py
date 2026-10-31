"""Perception helpers that use only drawer_observe payloads."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

ARM_MOUNT_XY = np.array([0.18, 0.48])


def _world_from_top(observation: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = observation["cameras"]["top"]
    depth_m = observation["depth_images"]["top"].astype(np.float64) / 1000.0
    calibration = observation["calibration"]["top"]
    rotation = np.asarray(calibration["rotation_camera_to_world"], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(calibration["position_m"], dtype=np.float64)
    fx, fy = calibration["fx"], calibration["fy"]
    cx, cy = calibration["cx"], calibration["cy"]
    height, width = depth_m.shape
    rows, cols = np.mgrid[0:height, 0:width]
    valid = (depth_m > 0.2) & (depth_m < 2.0)
    camera_points = np.stack(
        [
            (cols - cx) / fx * depth_m,
            (cy - rows) / fy * depth_m,
            -depth_m,
        ],
        axis=-1,
    )
    world = camera_points @ rotation.T + origin
    return rgb, depth_m, world, valid


def find_lower_handle(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    rgb, _depth, world, valid = _world_from_top(observation)
    handle = (
        valid
        & (rgb.max(axis=-1) < 50)
        & (world[:, :, 2] > 0.045)
        & (world[:, :, 2] < 0.095)
    )
    if int(handle.sum()) < 30:
        raise RuntimeError("handle not found in overhead view")
    points = world[handle]
    centroid = points.mean(axis=0)
    flat = points[:, :2] - centroid[:2]
    _, _, axes = np.linalg.svd(flat, full_matrices=False)
    normal = np.array([-axes[0, 1], axes[0, 0]])
    if float(np.dot(normal, ARM_MOUNT_XY - centroid[:2])) < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    return centroid, normal


def find_green_cube(observation: dict) -> np.ndarray:
    rgb, _depth, world, valid = _world_from_top(observation)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mask = (
        valid
        & (green > 100)
        & (green > red + 20)
        & (green > blue + 40)
        & (blue < 160)
        & (world[:, :, 2] > 0.004)
        & (world[:, :, 2] < 0.035)
    )
    labeled, count = ndimage.label(mask)
    best = None
    best_count = 0
    for index in range(1, count + 1):
        points = world[labeled == index]
        if len(points) < 40:
            continue
        span = points[:, :2].max(axis=0) - points[:, :2].min(axis=0)
        if float(span.max()) > 0.06:
            continue
        centroid = points.mean(axis=0)
        if float(centroid[1]) > 0.42 or float(centroid[1]) < 0.05:
            continue
        if len(points) > best_count:
            best_count = len(points)
            best = centroid
    if best is None:
        raise RuntimeError("green cube not found in overhead view")
    return best


def end_effector_xy(observation: dict) -> np.ndarray:
    """Approximate EE from forward kinematics is not in observe; use wrist unavailable.

    The step/observe payload includes joint positions. For this session the
    controller tracks EE by integrating commanded motion and by re-observing
    the overhead scene; callers should prefer measured handle/cube points.
    """
    raise NotImplementedError("use scene points from cameras, not a privileged EE site")
