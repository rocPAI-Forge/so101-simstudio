#!/usr/bin/env python3
"""Parse hz_bench eval logs into RESULTS.md. Drop episode 1 (compile warmup).

Logs default to repo outputs/eval/hz_bench/ (gitignored). This file writes
RESULTS.md next to itself under labs/lab01_pnp/hz_bench/.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
REPO = LAB_DIR.parents[2]
LOG_DIR = Path(os.environ.get("LAB01_HZ_BENCH_LOG", REPO / "outputs/eval/hz_bench"))
EP_RE = re.compile(
    r"Episode (\d+)/4: \S+ .* infer=([\d.]+) Hz  chunk=([\d.]+) Hz"
)
DONE_RE = re.compile(
    r"Eval complete: (\d+)/(\d+) success.*policy infer ([\d.]+) Hz.*"
    r"chunk ([\d.]+) Hz \((\d+) calls, mean ([\d.]+)ms\).*control compute ([\d.]+) Hz"
)

GPU_COL = {
    "act6": "infer",
    "act15": "infer",
    "smolvla": "chunk",
    "molmo6": "infer",
    "molmo15": "infer",
    "jepa": "infer",
    "groot": "chunk",
    "fastwam": "infer",
}


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    eps = []
    for m in EP_RE.finditer(text):
        eps.append(
            {
                "n": int(m.group(1)),
                "infer": float(m.group(2)),
                "chunk": float(m.group(3)),
            }
        )
    dm = DONE_RE.search(text)
    done = None
    if dm:
        done = {
            "ok": int(dm.group(1)),
            "n": int(dm.group(2)),
            "infer": float(dm.group(3)),
            "chunk": float(dm.group(4)),
            "chunk_calls": int(dm.group(5)),
            "chunk_ms": float(dm.group(6)),
            "control": float(dm.group(7)),
        }
    steady = [e for e in eps if e["n"] >= 2]
    return {"path": path.name, "eps": eps, "steady": steady, "done": done, "ok": "Eval complete" in text}


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def row(host: str, policy: str, rec: dict) -> str:
    metric = GPU_COL.get(policy, "infer")
    st = rec["steady"]
    gpu = median([e[metric] for e in st]) if st else None
    inf = median([e["infer"] for e in st]) if st else None
    ch = median([e["chunk"] for e in st]) if st else None
    d = rec["done"]
    ctrl = d["control"] if d else None
    status = "ok" if rec["ok"] else "partial"
    gpu_s = f"{gpu:.1f}" if gpu is not None else "—"
    inf_s = f"{inf:.1f}" if inf is not None else "—"
    ch_s = f"{ch:.1f}" if ch is not None else "—"
    ctrl_s = f"{ctrl:.1f}" if ctrl is not None else "—"
    ms = f"{1000.0 / gpu:.0f}" if gpu else "—"
    return (
        f"| {host} | {policy} | **{gpu_s}** ({metric}) | {ms} | "
        f"{inf_s} | {ch_s} | {ctrl_s} | {status} |"
    )


def main() -> None:
    logs = sorted(LOG_DIR.glob("halo_egl_*.log")) + sorted(LOG_DIR.glob("dorobot_osmesa_*.log"))
    parsed: list[tuple[str, str, dict]] = []
    for p in logs:
        name = p.stem  # halo_egl_act6
        parts = name.split("_", 2)
        if len(parts) < 3:
            continue
        host, _gl, policy = parts[0], parts[1], parts[2]
        parsed.append((host, policy, parse_log(p)))

    skips = []
    for s in (LOG_DIR / "halo_skip.txt", LOG_DIR / "dorobot_skip.txt"):
        if s.exists():
            skips.append(s.read_text())

    lines = [
        "# Lab 01 headless inference Hz (auto)",
        "",
        "Protocol: `render_window=false`, fixed cube, 4 episodes, **drop ep 1** (compile).",
        "halo = 8060S EGL. DORobot = MI300X OSMesa (control Hz is software render; trust GPU column).",
        "GPU column: sync `infer` / SmolVLA+GR00T `chunk` (`predict_action_chunk`). Not RTC queue pops.",
        "Sweep entry: `labs/lab01_pnp/hz_bench.cmd`. Logs: `outputs/eval/hz_bench/` (not committed).",
        "",
        "| Host | Policy | GPU Hz (metric) | mean ms | infer Hz | chunk Hz | control Hz | status |",
        "|------|--------|-----------------|--------:|---------:|---------:|-----------:|--------|",
    ]
    order = ["act6", "act15", "smolvla", "molmo6", "molmo15", "jepa", "groot", "fastwam"]
    parsed.sort(key=lambda t: (0 if t[0] == "halo" else 1, order.index(t[1]) if t[1] in order else 99))
    for host, policy, rec in parsed:
        lines.append(row(host, policy, rec))
    if skips:
        lines += ["", "## Skips", "", "```", *skips, "```"]
    halo_done = any(
        "DONE halo" in p.read_text() for p in LOG_DIR.glob("halo_progress.txt") if p.exists()
    )
    drob_done = any(
        "DONE dorobot" in p.read_text() for p in LOG_DIR.glob("dorobot_progress.txt") if p.exists()
    )
    lines += [
        "",
        f"Sweep complete: halo={'yes' if halo_done else 'no'}, dorobot={'yes' if drob_done else 'no'}.",
    ]
    out = LAB_DIR / "RESULTS.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()
