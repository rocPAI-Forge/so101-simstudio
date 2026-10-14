#!/usr/bin/env python3
"""Launch lerobot-train with the last N episodes repeated in the sampler.

LeRobot's episode sampler stores a boolean mask, so passing an episode index
twice does not raise its sampling rate. This launcher appends extra copies of
those episodes' frame ranges before training starts. It does not edit lerobot.
"""

from __future__ import annotations

import sys


def _take_flag(argv: list[str], name: str, default: int) -> tuple[list[str], int]:
    kept: list[str] = []
    value = default
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == name:
            value = int(argv[i + 1])
            i += 2
            continue
        if arg.startswith(name + "="):
            value = int(arg.split("=", 1)[1])
            i += 1
            continue
        kept.append(arg)
        i += 1
    return kept, value


def main() -> None:
    argv, last_n = _take_flag(sys.argv[1:], "--upweight-last", 0)
    argv, repeat = _take_flag(argv, "--upweight-repeat", 4)
    if last_n < 0 or repeat < 1:
        raise SystemExit("--upweight-last must be >= 0 and --upweight-repeat must be >= 1")
    sys.argv = [sys.argv[0], *argv]

    from lerobot.datasets.sampler import EpisodeAwareSampler

    original_init = EpisodeAwareSampler.__init__

    def upweighted_init(
        self,
        dataset_from_indices,
        dataset_to_indices,
        episode_indices_to_use=None,
        *args,
        **kwargs,
    ):
        from_indices = list(dataset_from_indices)
        to_indices = list(dataset_to_indices)
        if last_n and episode_indices_to_use is None:
            start = len(from_indices) - last_n
            if start < 0:
                raise RuntimeError(
                    f"Cannot upweight last {last_n} episodes; dataset has {len(from_indices)}"
                )
            extra_from = []
            extra_to = []
            for index in range(start, len(from_indices)):
                extra_from.extend([from_indices[index]] * (repeat - 1))
                extra_to.extend([to_indices[index]] * (repeat - 1))
            from_indices.extend(extra_from)
            to_indices.extend(extra_to)
            print(
                f"upweight last {last_n} episodes x{repeat} "
                f"(sampler ranges {len(from_indices)})",
                flush=True,
            )
        original_init(
            self,
            from_indices,
            to_indices,
            episode_indices_to_use,
            *args,
            **kwargs,
        )

    EpisodeAwareSampler.__init__ = upweighted_init

    from lerobot.scripts.lerobot_train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
