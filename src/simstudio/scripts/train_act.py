"""Lab 01 ACT/VLA training entry: optional 6-D state slice, then LeRobot train."""

from __future__ import annotations

import logging
import os

from simstudio.datasets.slice_obs_state import patch_lerobot_observation_state_dim


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def patch_vla_jepa_freeze_except_action_model() -> None:
    """Keep Qwen frozen and train only ``action_model`` (Lab 01 JEPA action-head FT)."""
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAModel

    orig = VLAJEPAModel.__init__

    def wrapped(self, config) -> None:  # type: ignore[no-untyped-def]
        orig(self, config)
        trainable = 0
        frozen = 0
        for name, param in self.named_parameters():
            if name.startswith("action_model"):
                param.requires_grad_(True)
                trainable += param.numel()
            else:
                param.requires_grad_(False)
                frozen += param.numel()
        logging.getLogger(__name__).info(
            "LAB01_JEPA_FREEZE_EXCEPT_ACTION: trainable action_model=%s frozen=%s",
            f"{trainable:,}",
            f"{frozen:,}",
        )

    VLAJEPAModel.__init__ = wrapped  # type: ignore[method-assign]


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

    if _truthy("LAB01_JEPA_FREEZE_EXCEPT_ACTION"):
        logging.basicConfig(level=logging.INFO)
        patch_vla_jepa_freeze_except_action_model()

    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
