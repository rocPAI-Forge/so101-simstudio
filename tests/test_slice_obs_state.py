import numpy as np
import torch

from simstudio.datasets.slice_obs_state import (
    apply_slice_to_metadata,
    slice_feature_spec,
    slice_stats_entry,
    slice_vector,
)


def test_slice_vector_torch_and_numpy():
    t = torch.arange(15.0)
    assert slice_vector(t, 6).tolist() == list(range(6))
    a = np.arange(15, dtype=np.float32)
    np.testing.assert_array_equal(slice_vector(a, 6), np.arange(6, dtype=np.float32))
    window = torch.arange(30.0).reshape(2, 15)
    assert slice_vector(window, 6).shape == (2, 6)


def test_slice_feature_and_stats():
    feat = {
        "dtype": "float32",
        "names": [f"d{i}" for i in range(15)],
        "shape": [15],
    }
    sliced = slice_feature_spec(feat, 6)
    assert sliced["shape"] == [6]
    assert sliced["names"] == [f"d{i}" for i in range(6)]
    stats = slice_stats_entry(
        {"mean": np.arange(15, dtype=np.float32), "std": np.ones(15, dtype=np.float32)},
        6,
    )
    assert stats["mean"].shape == (6,)


class _Meta:
    def __init__(self):
        self.features = {
            "observation.state": {
                "dtype": "float32",
                "names": [f"d{i}" for i in range(15)],
                "shape": [15],
            }
        }
        self.stats = {
            "observation.state": {"mean": np.arange(15, dtype=np.float32)},
        }


def test_apply_slice_to_metadata():
    meta = _Meta()
    apply_slice_to_metadata(meta, 6)
    assert meta.features["observation.state"]["shape"] == [6]
    assert meta.stats["observation.state"]["mean"].shape == (6,)
    apply_slice_to_metadata(meta, 6)  # idempotent when already 6
    assert meta.features["observation.state"]["shape"] == [6]
