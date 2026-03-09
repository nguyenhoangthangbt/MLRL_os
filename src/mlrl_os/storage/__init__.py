"""Storage backend abstraction for ML/RL OS.

Provides a unified StorageBackend protocol with file-system and PostgreSQL
implementations. The file backend wraps v0.1 registries; the Postgres backend
stores metadata in a database with Parquet files on disk.

Usage::

    from mlrl_os.storage import create_storage_backend

    backend = create_storage_backend(settings)
    meta = backend.register_dataset(raw_dataset, name="my-dataset")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mlrl_os.storage.file_backend import FileStorageBackend
from mlrl_os.storage.protocol import StorageBackend

if TYPE_CHECKING:
    from mlrl_os.config.defaults import MLRLSettings

logger = logging.getLogger(__name__)

__all__ = [
    "StorageBackend",
    "FileStorageBackend",
    "create_storage_backend",
]


def create_storage_backend(settings: MLRLSettings) -> StorageBackend:
    """Factory: create the appropriate storage backend from settings.

    Args:
        settings: Application settings. Uses ``storage_backend`` to select
            the backend type (``"file"`` or ``"postgres"``).

    Returns:
        A :class:`StorageBackend` instance.

    Raises:
        ValueError: If ``storage_backend`` is not a known value.
        ImportError: If ``"postgres"`` is selected but asyncpg is not installed.
    """
    backend_type = settings.storage_backend.lower()

    if backend_type == "file":
        return FileStorageBackend(settings)

    if backend_type == "postgres":
        from mlrl_os.storage.postgres_backend import PostgresStorageBackend

        return PostgresStorageBackend(settings)

    msg = (
        f"Unknown storage_backend: {settings.storage_backend!r}. "
        f"Must be 'file' or 'postgres'."
    )
    raise ValueError(msg)
