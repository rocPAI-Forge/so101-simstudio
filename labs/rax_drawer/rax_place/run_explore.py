"""Run the compiled controller on explore scenes and record successes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from labs.rax_drawer.rax_place.code.open_place_close.controller import main
from labs.rax_drawer.rax_place.tools import Session

HERE = Path(__file__).resolve().parent


def run(n: int, start: int = 0) -> dict:
    results = []
    with Session(split="explore", episode_limit_s=180.0) as ctx:
        # Advance the private layout cursor.
        for _ in range(start):
            ctx.reset(include_cameras=False)
        for i in range(n):
            try:
                ok = bool(main(ctx))
            except Exception as exc:  # noqa: BLE001 - agent run must continue
                ok = False
                err = f"{type(exc).__name__}: {exc}"
            else:
                err = ""
            # main() already called reset once; Session advances layout each reset.
            # When main fails before reset completion, force a clean boundary.
            if ctx.backend.env.terminated is False:
                pass
            results.append({"i": start + i, "success": ok, "error": err})
            print(f"explore {start + i}: success={ok} {err}", flush=True)
    out = HERE / "logs" / "explore_self_test.json"
    out.write_text(json.dumps({"n": n, "start": start, "results": results}, indent=2))
    ok_n = sum(1 for row in results if row["success"])
    print(f"EXPLORE_SELF_TEST {ok_n} / {n}")
    return {"ok": ok_n, "n": n, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    run(args.n, args.start)
