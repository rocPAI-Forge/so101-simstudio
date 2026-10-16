"""Benchmark an external VLA backend and enforce the RTC capacity gate."""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from simstudio.common.constants import DEFAULT_HOME_POSITION, JOINT_NAMES
from simstudio.policies.external_vla import (
    DeploymentManifest,
    InProcessBackend,
    ObservationFrame,
    SafeBackend,
    SidecarClient,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--backend", choices=("inprocess", "sidecar"), default="sidecar")
    parser.add_argument("--endpoint", default="http://127.0.0.1:10086")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        raise SystemExit("--iterations must be positive and --warmup cannot be negative")

    manifest = DeploymentManifest.load(args.manifest)
    raw_backend = (
        InProcessBackend(manifest)
        if args.backend == "inprocess"
        else SidecarClient(
            args.endpoint,
            timeout_s=args.timeout_s,
            jpeg_quality=args.jpeg_quality,
        )
    )
    backend = SafeBackend(
        raw_backend,
        manifest,
        stale_after_s=max(args.timeout_s, 1.0),
    )
    blank = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
    observation = ObservationFrame(
        task="External VLA latency benchmark.",
        joint_positions={name: DEFAULT_HOME_POSITION[name] for name in JOINT_NAMES},
        images={name: blank for name in manifest.required_cameras},
    )
    backend.reset_episode("benchmark")
    try:
        for _ in range(args.warmup):
            backend.predict_chunk(observation)
        latencies_ms: list[float] = []
        reported: dict[str, list[float]] = {}
        for _ in range(args.iterations):
            started = time.perf_counter()
            chunk = backend.predict_chunk(observation)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            for name, value in chunk.timings_ms.items():
                reported.setdefault(name, []).append(float(value))
    finally:
        backend.close()

    mean_ms = statistics.fmean(latencies_ms)
    p95_ms = sorted(latencies_ms)[max(0, int(0.95 * len(latencies_ms)) - 1)]
    chunk_hz = 1000.0 / mean_ms
    capacity_hz = chunk_hz * manifest.execution_horizon
    print(
        f"external-vla: mean={mean_ms:.1f}ms p95={p95_ms:.1f}ms "
        f"chunk={chunk_hz:.2f}Hz execution_horizon={manifest.execution_horizon} "
        f"capacity={capacity_hz:.2f}Hz target={manifest.control_fps:.2f}Hz"
    )
    for name, values in sorted(reported.items()):
        print(f"  {name}: mean={statistics.fmean(values):.1f}ms")
    if capacity_hz < manifest.control_fps:
        raise SystemExit("FAIL: chunk_hz * execution_horizon is below the requested control rate")
    print("PASS: RTC capacity gate")


if __name__ == "__main__":
    main()
