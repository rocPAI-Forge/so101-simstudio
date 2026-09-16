"""Run Lab 01's 6-D joint actions inside FastWAM's pretrained 7-D action pathway.

``lerobot/fastwam_base`` ships **no** ``action_encoder``, action ``head`` or
``proprio_encoder`` — LeRobot logs them as missing keys and they start from a
fresh init, so a 6-D fine-tune trains the whole action in/out pathway from
scratch on top of a trained 1B action trunk. The LIBERO fine-tune
``ZibinDong/fastwam_libero_uncond_2cam224`` does ship them, at 7-D action and
8-D proprio.

Keeping the checkpoint's dims and padding Lab 01's 6 joints into the leading
channels inherits that trained pathway. Padding is applied *after*
normalization, so a padded channel sits at 0 (the dataset mean in normalized
space) and is identical at train and rollout time. Predictions are sliced back
to the real channels before the unnormalizer sees them.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

OBS_STATE = "observation.state"


def _pad_last_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    current = int(tensor.shape[-1])
    if current >= dim:
        return tensor
    pad = torch.zeros(
        *tensor.shape[:-1],
        dim - current,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=-1)


def _pad_batch(batch: dict[str, Any], action_dim: int, proprio_dim: int | None) -> dict[str, Any]:
    out = dict(batch)
    action = out.get("action")
    if isinstance(action, torch.Tensor):
        out["action"] = _pad_last_dim(action, action_dim)
    state = out.get(OBS_STATE)
    if proprio_dim is not None and isinstance(state, torch.Tensor):
        out[OBS_STATE] = _pad_last_dim(state, proprio_dim)
    return out


_PAD_PATCHED = False


def patch_fastwam_pad_action_channels(real_action_dim: int = 6) -> None:
    """Pad Lab 01 channels up to the checkpoint's action/proprio dims.

    Must run **before** ``FastWAMConfig`` is constructed: saved checkpoints keep
    ``action_dim=7`` / ``proprio_dim=8`` but ``output_features.action`` is 6-D,
    so ``validate_features`` otherwise raises during eval config load.

    Args:
        real_action_dim: Channels the robot actually consumes (Lab 01: 6 joints).
            Predicted chunks are sliced back to this width.
    """
    global _PAD_PATCHED
    if _PAD_PATCHED:
        return
    _PAD_PATCHED = True

    from lerobot.configs.types import FeatureType
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
    from lerobot.utils.constants import ACTION

    orig_validate = FastWAMConfig.validate_features

    def validate_with_padded_dims(self) -> None:  # type: ignore[no-untyped-def]
        """Let the dataset declare 6-D action while the model stays at 7-D."""
        saved_action, saved_proprio = self.action_dim, self.proprio_dim
        action_ft = (self.output_features or {}).get(ACTION)
        state_ft = (self.input_features or {}).get(OBS_STATE)
        if action_ft is not None and action_ft.type is FeatureType.ACTION:
            self.action_dim = int(action_ft.shape[-1])
        if state_ft is not None and saved_proprio is not None:
            self.proprio_dim = int(state_ft.shape[-1])
        try:
            orig_validate(self)
        finally:
            self.action_dim, self.proprio_dim = saved_action, saved_proprio

    FastWAMConfig.validate_features = validate_with_padded_dims  # type: ignore[method-assign]

    orig_batch = FastWAMPolicy._batch_to_training_sample

    def batch_with_padding(self, batch):  # type: ignore[no-untyped-def]
        return orig_batch(self, _pad_batch(batch, self.config.action_dim, self.config.proprio_dim))

    FastWAMPolicy._batch_to_training_sample = batch_with_padding  # type: ignore[method-assign]

    orig_predict = FastWAMPolicy.predict_action_chunk

    def predict_then_slice(self, batch, **kwargs):  # type: ignore[no-untyped-def]
        padded = _pad_batch(batch, self.config.action_dim, self.config.proprio_dim)
        chunk = orig_predict(self, padded, **kwargs)
        return chunk[..., :real_action_dim]

    FastWAMPolicy.predict_action_chunk = predict_then_slice  # type: ignore[method-assign]

def patch_fastwam_unfreeze_last_video_layers(n_layers: int) -> None:
    """Keep the 5B video expert frozen except the last ``n_layers`` MoT blocks.

    Call after ``freeze_video_expert=true``. Extra trainable params are roughly
    ``n_layers / 30`` of the video expert (~0.17B per layer).
    """
    if n_layers <= 0:
        return
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

    orig = FastWAMPolicy.__init__

    def wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        orig(self, *args, **kwargs)
        mot = getattr(getattr(self, "model", None), "mot", None)
        layers = getattr(mot, "layers", None) if mot is not None else None
        if not layers:
            logger.warning("FastWAM unfreeze skipped: no MoT layers")
            return
        n = min(int(n_layers), len(layers))
        thawed = 0
        for layer in layers[-n:]:
            blocks = getattr(layer, "blocks", None)
            if blocks is None or "video" not in blocks:
                continue
            block = blocks["video"]
            block.requires_grad_(True)
            thawed += sum(p.numel() for p in block.parameters())
        logger.info(
            "FastWAM unfroze last %d/%d video MoT layers (%s params)",
            n,
            len(layers),
            f"{thawed:,}",
        )

    FastWAMPolicy.__init__ = wrapped  # type: ignore[method-assign]
