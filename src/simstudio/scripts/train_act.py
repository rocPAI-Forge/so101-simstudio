"""Lab 01 ACT/VLA training entry: optional 6-D state slice, then LeRobot train."""

from __future__ import annotations

import logging
import os
import sys

from simstudio.datasets.slice_obs_state import patch_lerobot_observation_state_dim


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def patch_pi05_train_vision_freeze_lm() -> None:
    """Train the pi05 vision tower, projector, and action expert. Keep the language model frozen.

    ``train_expert_only`` freezes all of PaliGemma. This patch runs after that setup and
    re-opens only the vision path. The language model stays in eval mode.
    """
    from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel

    orig_init = PaliGemmaWithExpertModel.__init__
    orig_train = PaliGemmaWithExpertModel.train

    def _vision_param(name: str) -> bool:
        return "vision_tower" in name or "multi_modal_projector" in name

    def _apply(self) -> None:
        vision = frozen = 0
        for name, param in self.paligemma.named_parameters():
            keep = _vision_param(name)
            param.requires_grad_(keep)
            if keep:
                vision += param.numel()
            else:
                frozen += param.numel()
        expert = sum(p.numel() for p in self.gemma_expert.parameters() if p.requires_grad)
        logging.getLogger(__name__).info(
            "LAB01_PI05_TRAIN_VISION: vision=%s frozen_lm=%s action_expert=%s",
            f"{vision:,}",
            f"{frozen:,}",
            f"{expert:,}",
        )

    def wrapped_init(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        orig_init(self, *args, **kwargs)
        self._lab01_train_vision = True
        _apply(self)

    def wrapped_train(self, mode: bool = True) -> None:  # type: ignore[no-untyped-def]
        orig_train(self, mode)
        if mode and getattr(self, "_lab01_train_vision", False):
            self.paligemma.eval()
            self.paligemma.model.vision_tower.train()
            projector = getattr(self.paligemma.model, "multi_modal_projector", None)
            if projector is not None:
                projector.train()

    PaliGemmaWithExpertModel.__init__ = wrapped_init  # type: ignore[method-assign]
    PaliGemmaWithExpertModel.train = wrapped_train  # type: ignore[method-assign]


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


def patch_fastwam_dit_configs_for_action_dim() -> None:
    """Rebuild FastWAM DiT configs when CLI ``action_dim`` disagrees with the Hub JSON.

    ``lerobot/fastwam_base`` stores ``video_dit_config.action_dim=7``. Overriding
    ``--policy.action_dim=6`` alone leaves the nested DiT dict at 7 and the
    action encoder/head stay 7-D. Rebuild so 6-DoF Lab 01 joints load with the
    existing shape-mismatch reinit path.
    """
    from lerobot.policies.fastwam.configuration_fastwam import (
        FastWAMConfig,
        default_action_dit_config,
        default_video_dit_config,
    )

    orig = FastWAMConfig.__post_init__

    def wrapped(self) -> None:  # type: ignore[no-untyped-def]
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.utils.constants import ACTION, OBS_STATE

        vd = self.video_dit_config or {}
        ckpt_dim = int(vd.get("action_dim", self.action_dim))
        if ckpt_dim != int(self.action_dim):
            logging.getLogger(__name__).info(
                "Rebuilding FastWAM DiT configs for action_dim=%s (checkpoint nested dim=%s)",
                self.action_dim,
                ckpt_dim,
            )
            self.video_dit_config = default_video_dit_config(self.action_dim)
            self.action_dit_config = default_action_dit_config(self.action_dim)
        out = dict(self.output_features or {})
        out[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        self.output_features = out
        if self.proprio_dim is not None:
            inputs = dict(self.input_features or {})
            inputs[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.proprio_dim,))
            self.input_features = inputs
        orig(self)

    FastWAMConfig.__post_init__ = wrapped  # type: ignore[method-assign]

    orig_meta = FastWAMConfig.set_dataset_feature_metadata

    def wrapped_meta(self, dataset_features):  # type: ignore[no-untyped-def]
        keep = {
            "observation.images.camera_top",
            "observation.images.camera_wrist",
            "observation.images.image",
            "observation.images.image2",
        }
        filtered = {}
        rename = {
            "observation.images.camera_top": "observation.images.image",
            "observation.images.camera_wrist": "observation.images.image2",
        }
        for key, value in dataset_features.items():
            if str(key).startswith("observation.images.") and key not in keep and key not in rename:
                continue
            filtered[rename.get(key, key)] = value
        logging.getLogger(__name__).info(
            "FastWAM dataset cameras kept: %s",
            [k for k in filtered if str(k).startswith("observation.images.")],
        )
        orig_meta(self, filtered)

    FastWAMConfig.set_dataset_feature_metadata = wrapped_meta  # type: ignore[method-assign]

    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

    orig_batch = FastWAMPolicy._batch_to_training_sample
    keep_cams = {
        "observation.images.image",
        "observation.images.image2",
        "observation.images.camera_top",
        "observation.images.camera_wrist",
    }

    def wrapped_batch(self, batch):  # type: ignore[no-untyped-def]
        batch = dict(batch)
        for key in list(batch):
            if not str(key).startswith("observation.images."):
                continue
            base = key[: -len("_is_pad")] if str(key).endswith("_is_pad") else key
            if base not in keep_cams:
                batch.pop(key)
        return orig_batch(self, batch)

    FastWAMPolicy._batch_to_training_sample = wrapped_batch  # type: ignore[method-assign]


def patch_lingbot_action_feature_from_used_channels() -> None:
    """Keep ``output_features[action]`` aligned with ``used_action_channel_ids``.

    The LIBERO Hub config stores action shape 7. Lab 01 is 6-D joints; if we only
    override ``used_action_channel_ids``, ``validate_features`` leaves the 7-D
    checkpoint feature in place and the dataloader dies on a shape mismatch.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
    from lerobot.utils.constants import ACTION

    orig = LingBotVAConfig.validate_features

    def wrapped(self) -> None:  # type: ignore[no-untyped-def]
        n_used = len(self.used_action_channel_ids)
        self.output_features = dict(self.output_features or {})
        self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(n_used,))
        orig(self)

    LingBotVAConfig.validate_features = wrapped  # type: ignore[method-assign]


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

    if _truthy("LAB01_PI05_TRAIN_VISION"):
        logging.basicConfig(level=logging.INFO)
        patch_pi05_train_vision_freeze_lm()

    if _truthy("LAB01_JEPA_FREEZE_EXCEPT_ACTION"):
        logging.basicConfig(level=logging.INFO)
        patch_vla_jepa_freeze_except_action_model()

    if _truthy("LAB01_PATCH_FASTWAM_DIT") or any("fastwam" in arg.lower() for arg in sys.argv):
        logging.basicConfig(level=logging.INFO)
        patch_fastwam_dit_configs_for_action_dim()

    if _truthy("LAB01_FASTWAM_PAD_ACTION"):
        from simstudio.policies.fastwam_pad import patch_fastwam_pad_action_channels

        logging.basicConfig(level=logging.INFO)
        patch_fastwam_pad_action_channels(int(os.environ.get("LAB01_FASTWAM_REAL_ACTION_DIM", "6")))

    raw_unfreeze = os.environ.get("LAB01_FASTWAM_UNFREEZE_VIDEO_LAST", "").strip()
    if raw_unfreeze:
        from simstudio.policies.fastwam_pad import patch_fastwam_unfreeze_last_video_layers

        logging.basicConfig(level=logging.INFO)
        patch_fastwam_unfreeze_last_video_layers(int(raw_unfreeze))

    if any("lingbot" in arg.lower() for arg in sys.argv):
        logging.basicConfig(level=logging.INFO)
        patch_lingbot_action_feature_from_used_channels()

    if any("groot" in arg.lower() for arg in sys.argv) or _truthy("LAB01_GROOT"):
        # Register policy.type=groot. GrootConfig.__post_init__ rejects N1.5 paths.
        from lerobot.policies.groot.configuration_groot import GrootConfig  # noqa: F401
        from simstudio.policies.groot_n17 import patch_groot_n17_vlm_processor_source

        logging.basicConfig(level=logging.INFO)
        patch_groot_n17_vlm_processor_source()

    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
