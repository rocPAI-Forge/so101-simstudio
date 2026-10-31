"""One-shot locked evaluation. Do not use results to retune the controller."""

from __future__ import annotations

import json
from pathlib import Path

from labs.rax_drawer.rax_place.code.open_place_close.controller import main
from labs.rax_drawer.rax_place.tools import Session

HERE = Path(__file__).resolve().parent
FREEZE_MARK = HERE / "session" / "CONTROLLER_FROZEN"


def run(*, round_id: int = 1) -> dict:
    mark = FREEZE_MARK if round_id == 1 else HERE / "session" / f"CONTROLLER_FROZEN_{round_id}"
    if not mark.exists():
        raise SystemExit(f"Refuse locked test until {mark.name} exists")
    suffix = "" if round_id == 1 else f"_{round_id}"
    results = []
    with Session(split="test", episode_limit_s=180.0, audit_path=HERE / "logs" / f"mcp_test{suffix}.jsonl") as ctx:
        for i in range(len(ctx.scenes)):
            try:
                ok = bool(main(ctx))
                err = ""
            except Exception as exc:  # noqa: BLE001
                ok = False
                err = f"{type(exc).__name__}: {exc}"
            results.append({"i": i, "success": ok, "error": err})
            print(f"test {i}: success={ok} {err}", flush=True)
    ok_n = sum(1 for row in results if row["success"])
    out = {
        "n": len(results),
        "successes": ok_n,
        "rate": ok_n / max(len(results), 1),
        "results": results,
        "frozen_at": mark.read_text().strip(),
        "round": round_id,
    }
    path = HERE / "logs" / f"locked_test{suffix}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"LOCKED_TEST round {round_id} {ok_n} / {len(results)}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()
    run(round_id=args.round)
