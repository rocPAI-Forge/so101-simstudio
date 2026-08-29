"""Lab 01 ACT training entry: optional 6-D ``observation.state`` slice, then LeRobot train."""

from __future__ import annotations

import logging
import os

from simstudio.datasets.slice_obs_state import patch_lerobot_observation_state_dim


def main() -> None:
    raw = (
        os.environ.get("LAB01_STATE_DIM", "").strip()
        or os.environ.get("LAB01_ACT_STATE_DIM", "").strip()
    )
    if raw:
        dim = int(raw)
        if dim <= 0:
            raise SystemExit(f"LAB01_STATE_DIM/LAB01_ACT_STATE_DIM must be a positive int, got {raw!r}")
        logging.basicConfig(level=logging.INFO)
        patch_lerobot_observation_state_dim(dim)

    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
