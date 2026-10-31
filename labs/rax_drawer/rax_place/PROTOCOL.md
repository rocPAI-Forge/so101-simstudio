# RAX protocol for open → place → close

## Goal

Reach at least **10 / 20** success on the locked test set for the long-horizon task
(open drawer, place green cube in tray, close drawer), following RAX:

1. Explore only on `manifests/explore.json`.
2. Control uses only tool observations (joints, calibration, cameras, overhead depth).
3. Compile a program from successful experience.
4. Score the frozen program once on `manifests/test.json`.

## What this session is

- Official Claude Code harness credentials are **not** available on this host.
- Cursor acts as the coding agent through `tools.Session`, which owns the robot and
  hides cabinet layout, drawer joint, cube pose, handle pose, and contacts.
- Every tool call is appended to `logs/mcp.jsonl`.
- Notes go in `memory/`. The compiled program is `code/open_place_close/controller.py`.

## Contamination note

The same host already contains a hand-tuned `place_close.py` from an earlier
non-RAX debugging pass. This session must not import that module into the
compiled controller. Prior human knowledge is a risk; the audit log and the
rule “never open test for tuning” still apply.

## Budget

- Explore episodes: target 16–32 before compile.
- Explore self-test: at least 8 scenes after compile.
- Locked test: exactly one frozen 20-scene run after freeze.
