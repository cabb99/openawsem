"""``esm`` backend (experimental): ESMFold structure or ESM distance distribution."""
from __future__ import annotations

from openawsem.memory.generators.base import FragmentBackend, Registry


@Registry.register("esm", experimental=True)
class Esm(FragmentBackend):
    """ESMFold structure or ESM distance distribution -> memory."""
