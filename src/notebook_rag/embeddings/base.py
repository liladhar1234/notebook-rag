"""Embedder protocols — all embedders must satisfy one of these interfaces."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from qdrant_client.models import SparseVector


@runtime_checkable
class DenseEmbedderProtocol(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class SparseEmbedderProtocol(Protocol):
    def encode(self, texts: list[str]) -> list[SparseVector]: ...
