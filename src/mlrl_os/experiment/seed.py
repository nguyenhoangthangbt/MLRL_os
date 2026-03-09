"""Deterministic seeded RNG utilities.

Same pattern as SimOS: each component gets an independent Random
seeded with seed_hash(component_name, global_seed).
Same global seed = identical results. Changing one component does not affect others.
"""

from __future__ import annotations

import hashlib
import random


def seed_hash(name: str, global_seed: int) -> int:
    """Compute a deterministic seed for a named component.

    Args:
        name: Component name (e.g., "cv_split", "lightgbm_train").
        global_seed: The experiment's global seed.

    Returns:
        Deterministic integer seed derived from name + global_seed.
    """
    data = f"{name}:{global_seed}".encode()
    digest = hashlib.sha256(data).digest()[:4]
    return int.from_bytes(digest, byteorder="big")


def make_rng(name: str, global_seed: int) -> random.Random:
    """Create a seeded Random instance for a named component.

    Args:
        name: Component name.
        global_seed: The experiment's global seed.

    Returns:
        A random.Random instance seeded deterministically.
    """
    return random.Random(seed_hash(name, global_seed))
