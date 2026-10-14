"""GR00T N1.7 Lab 01 helpers (gated Cosmos tokenizer → Qwen3-VL processor files).

LeRobot's N1.7 VLM encode step loads ``nvidia/Cosmos-Reason2-2B`` for the
tokenizer / Qwen3-VL processor. That Hub repo is gated; ``nvidia/GR00T-N1.7-3B``
weights themselves are not. Cosmos is a Qwen3-VL-2B-Instruct layout (LeRobot
already hard-codes that config for the backbone). We point the *processor*
at public ``Qwen/Qwen3-VL-2B-Instruct`` tokenizer assets so smoke/train can
run without NVIDIA Cosmos access. Set ``LAB01_GROOT_PROCESSOR_MODEL`` to a
local snapshot or ``nvidia/Cosmos-Reason2-2B`` if you have been granted access.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_COSMOS_IDS = frozenset(
    {
        "nvidia/Cosmos-Reason2-2B",
        "nvidia/Cosmos-Reason2-2B/",
    }
)
_DEFAULT_PUBLIC_PROCESSOR = "Qwen/Qwen3-VL-2B-Instruct"
_TOKENIZER_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "tokenizer*",
    "vocab*",
    "merges*",
    "preprocessor*",
    "chat_template*",
    "video_preprocessor*",
)


def _is_cosmos_id(model_name: str) -> bool:
    return str(model_name).rstrip("/") in {s.rstrip("/") for s in _COSMOS_IDS}


def resolve_groot_n17_processor_model() -> str:
    """Return a local path or Hub id the transformers processor can load."""
    explicit = os.environ.get("LAB01_GROOT_PROCESSOR_MODEL", "").strip()
    repo = explicit or _DEFAULT_PUBLIC_PROCESSOR
    if Path(repo).exists():
        return repo

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        logger.info("Prefetching GR00T N1.7 processor files from %s", repo)
        return snapshot_download(repo, allow_patterns=list(_TOKENIZER_PATTERNS))


def patch_groot_n17_vlm_processor_source() -> None:
    """Redirect gated Cosmos tokenizer loads to ``LAB01_GROOT_PROCESSOR_MODEL``."""
    from lerobot.policies.groot import processor_groot as pg

    resolved = resolve_groot_n17_processor_model()
    orig = pg._build_n1_7_processor
    pg.GROOT_N1_7_BACKBONE_MODEL = resolved

    def wrapped(model_name: str = resolved):  # type: ignore[no-untyped-def]
        path = Path(str(model_name)) if model_name else None
        missing_local = bool(path and path.is_absolute() and not path.exists())
        if _is_cosmos_id(model_name) or not model_name or missing_local:
            logger.info(
                "GR00T N1.7 VLM processor: %s (requested %s)",
                resolved,
                model_name or "<empty>",
            )
            model_name = resolved
        return orig(model_name)

    pg._build_n1_7_processor = wrapped  # type: ignore[method-assign]
    logger.info("LAB01_GROOT processor source=%s", resolved)
