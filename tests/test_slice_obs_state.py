import pickle

import numpy as np
import torch

from simstudio.datasets.slice_obs_state import (
    _ObservationStateSliceWorkerInit,
    apply_slice_to_metadata,
    install_observation_state_slice,
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


def test_install_slices_getitem_and_getitems():
    class _DS:
        def __init__(self):
            self.meta = _Meta()

        def __getitem__(self, idx):
            return {"observation.state": torch.arange(15.0), "index": idx}

        def __getitems__(self, indices):
            return [{"observation.state": torch.arange(15.0), "index": i} for i in indices]

    install_observation_state_slice(_DS, 6)
    row = _DS()[3]
    assert row["observation.state"].tolist() == list(range(6))
    assert row["index"] == 3
    assert _DS().meta.features["observation.state"]["shape"] == [6]
    batch = _DS().__getitems__([1, 2])
    assert [item["index"] for item in batch] == [1, 2]
    assert batch[0]["observation.state"].shape == (6,)


def test_worker_init_survives_spawn_pickle():
    hook = _ObservationStateSliceWorkerInit(6, None)
    restored = pickle.loads(pickle.dumps(hook))
    assert restored.dim == 6
    assert restored.previous is None


def test_apply_slice_to_metadata():
    meta = _Meta()
    apply_slice_to_metadata(meta, 6)
    assert meta.features["observation.state"]["shape"] == [6]
    assert meta.stats["observation.state"]["mean"].shape == (6,)
    apply_slice_to_metadata(meta, 6)  # idempotent when already 6
    assert meta.features["observation.state"]["shape"] == [6]
