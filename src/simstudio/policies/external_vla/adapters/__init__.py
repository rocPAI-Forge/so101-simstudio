"""Reference adapters for the external VLA inference contract."""

from .base import VLAAdapter, load_adapter
from .fake import FakeAdapter

__all__ = ["FakeAdapter", "VLAAdapter", "load_adapter"]
