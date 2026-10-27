"""Slice Lab 01 ``observation.state`` to the first N dims (joint positions).

SimStudio records 15-D state (6 pos + 6 vel + 3 EE). Official real SO-101
LeRobot datasets are 6-D joint positions. Training can keep the on-disk 15-D
parquet and slice at load time so the policy / normalizer see 6-D only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

OBS_STATE = "observation.state"


def slice_vector(value: Any, dim: int) -> Any:
    """Keep the last-axis prefix of a 1-D feature (numpy / torch / list)."""
    if value is None:
        return value
    if isinstance(value, torch.Tensor):
        return value[..., :dim]
    if isinstance(value, np.ndarray):
        return value[..., :dim]
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        return arr[..., :dim]
    return value


def slice_feature_spec(feat: dict[str, Any], dim: int) -> dict[str, Any]:
    """Return a copy of a LeRobot feature dict sliced to ``dim``."""
    out = dict(feat)
    names = feat.get("names")
    if names:
        out["names"] = list(names)[:dim]
    shape = feat.get("shape")
    if shape:
        out["shape"] = [dim] if len(shape) == 1 else [*list(shape)[:-1], dim]
    return out


def slice_stats_entry(stats: dict[str, Any], dim: int) -> dict[str, Any]:
    return {key: slice_vector(val, dim) for key, val in stats.items()}


def apply_slice_to_metadata(meta: Any, dim: int) -> None:
    """Mutate in-memory dataset metadata so the policy infers a ``dim``-D state."""
    features = meta.features
    feat = features.get(OBS_STATE)
    if feat is None:
        return
    current = int(feat.get("shape", [0])[-1]) if feat.get("shape") else 0
    if current == 0 or current <= dim:
        return
    features[OBS_STATE] = slice_feature_spec(feat, dim)
    stats = getattr(meta, "stats", None)
    if isinstance(stats, dict) and OBS_STATE in stats:
        stats[OBS_STATE] = slice_stats_entry(stats[OBS_STATE], dim)
    logger.info(
        "Sliced observation.state metadata from %d to %d dims (on-disk dataset unchanged)",
        current,
        dim,
    )


def install_observation_state_slice(dataset_cls: type, dim: int) -> None:
    """Slice ``observation.state`` on a dataset class after rows are loaded.

    Metadata is sliced **after** ``__init__``: HuggingFace needs the on-disk
    schema to cast columns. Policy construction reads ``dataset.meta``
    afterwards, so it still sees ``dim``.

    Newer LeRobot loads training batches through ``__getitems__``, which does
    not call ``__getitem__``. Wrap both when ``__getitems__`` exists.
    """
    if dim <= 0:
        return

    orig_init = dataset_cls.__init__
    orig_getitem = dataset_cls.__getitem__

    def init_then_slice(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        apply_slice_to_metadata(self.meta, dim)

    def _slice_item(item: dict) -> dict:
        if isinstance(item, dict) and OBS_STATE in item:
            item[OBS_STATE] = slice_vector(item[OBS_STATE], dim)
        return item

    def getitem_and_slice(self, idx):
        return _slice_item(orig_getitem(self, idx))

    dataset_cls.__init__ = init_then_slice  # type: ignore[method-assign]
    dataset_cls.__getitem__ = getitem_and_slice  # type: ignore[method-assign]

    orig_getitems = getattr(dataset_cls, "__getitems__", None)
    if orig_getitems is not None:

        def getitems_and_slice(self, indices):
            return [_slice_item(item) for item in orig_getitems(self, indices)]

        dataset_cls.__getitems__ = getitems_and_slice  # type: ignore[method-assign]


_dataloader_slice_dim: int | None = None


class _ObservationStateSliceWorkerInit:
    """Re-apply the slice inside a DataLoader worker.

    LeRobot defaults to ``spawn``. A nested function cannot be pickled across
    that boundary. Workers also re-import ``LeRobotDataset`` and do not see a
    monkeypatch done only in the parent, so batches stay 15-D while the
    normalizer stats were sliced to ``dim``.
    """

    def __init__(self, dim: int, previous: Any | None) -> None:
        self.dim = dim
        self.previous = previous

    def __call__(self, worker_id: int) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        install_observation_state_slice(LeRobotDataset, self.dim)
        if self.previous is not None:
            self.previous(worker_id)


def _install_dataloader_worker_slice(dim: int) -> None:
    """Attach the picklable worker hook to later DataLoader construction."""
    global _dataloader_slice_dim
    if _dataloader_slice_dim == dim:
        return
    _dataloader_slice_dim = dim

    from torch.utils.data import DataLoader

    orig_init = DataLoader.__init__

    def init_with_slice(self, *args, **kwargs):
        num_workers = kwargs.get("num_workers", 0)
        if num_workers:
            kwargs["worker_init_fn"] = _ObservationStateSliceWorkerInit(
                dim,
                kwargs.get("worker_init_fn"),
            )
        orig_init(self, *args, **kwargs)

    DataLoader.__init__ = init_with_slice  # type: ignore[method-assign]


def patch_lerobot_observation_state_dim(dim: int) -> None:
    """Monkeypatch LeRobot dataset load so training sees a state prefix.

    Call this before constructing the dataset (i.e. before ``lerobot-train``).
    Does not rewrite parquet or videos.
    """
    if dim <= 0:
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    install_observation_state_slice(LeRobotDataset, dim)
    _install_dataloader_worker_slice(dim)
    logger.info("Will slice observation.state to %d dims after parquet load", dim)
