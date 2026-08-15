"""E4 — Kauffman's binary polymer model.

The canonical setting for the closure phase transition (Kauffman 1986; Hordijk &
Steel 2004). It is the substrate `DESIGN_abiogenesis.md` §6a's premise 1 was
established in, which is why premise 2 must be measured here and not on E1:
ReactionAtlas carries reactions with DFT barriers but **no catalysis relation**, and
exploitability is defined through catalysis.

Molecules are binary strings of length 1..`max_len`. The food set is every string of
length <= `food_len`. Reactions are **ligations** `a + b -> ab` for every ordered pair
whose concatenation is within `max_len`; the cleavage direction is omitted from the
primary model so that "F-generated" keeps its usual build-up-from-food reading, and
because including it makes every product trivially reachable.

Catalysis is assigned independently: each (molecule, reaction) pair is a catalysis
edge with probability `p`. That uniform assignment is the model's defining
simplification and also its main limitation — real catalysis is structured, which is
what ensembles E5/E6 exist to test.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product as _product
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class BinaryPolymerNetwork:
    """A generated binary-polymer chemistry.

    `reactions[i] = (a, b, ab)` as molecule indices. `catalysts[i]` is the set of
    molecule indices catalysing reaction `i`.
    """

    molecules: tuple[str, ...]
    food: frozenset[int]
    reactions: tuple[tuple[int, int, int], ...]
    catalysts: tuple[frozenset[int], ...]
    p: float
    max_len: int
    food_len: int

    @property
    def n_molecules(self) -> int:
        return len(self.molecules)

    @property
    def n_reactions(self) -> int:
        return len(self.reactions)

    @property
    def mean_catalysed_per_molecule(self) -> float:
        """Reactions catalysed per molecule — the model's natural control variable.

        Kauffman's transition is usually located in this quantity rather than in `p`
        directly, because `p` alone is not comparable across network sizes.
        """
        if not self.molecules:
            return 0.0
        return sum(len(c) for c in self.catalysts) / len(self.molecules)


def _strings(max_len: int) -> Iterator[str]:
    for n in range(1, max_len + 1):
        for bits in _product("01", repeat=n):
            yield "".join(bits)


def binary_polymer(max_len: int = 8, food_len: int = 2, p: float = 1e-3,
                   rng: np.random.Generator | None = None) -> BinaryPolymerNetwork:
    """Generate one binary-polymer network with catalysis at probability `p`."""
    if max_len < 2:
        raise ValueError(f"max_len must be at least 2, got {max_len}")
    if not 0 <= food_len < max_len:
        raise ValueError(f"food_len must be in [0, max_len), got {food_len}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p is a probability, got {p}")
    rng = rng or np.random.default_rng()

    molecules = tuple(_strings(max_len))
    index = {m: i for i, m in enumerate(molecules)}
    food = frozenset(i for i, m in enumerate(molecules) if len(m) <= food_len)

    reactions = tuple(
        (index[a], index[b], index[a + b])
        for a in molecules for b in molecules if len(a) + len(b) <= max_len
    )

    # Sparse Bernoulli draw: sampling the number of catalysts per reaction and then
    # which ones is O(edges) rather than O(molecules x reactions), which matters --
    # the dense product is ~10^8 at max_len=10.
    n_mol = len(molecules)
    counts = rng.binomial(n_mol, p, size=len(reactions))
    catalysts = tuple(
        frozenset(rng.choice(n_mol, size=int(k), replace=False).tolist()) if k else
        frozenset()
        for k in counts
    )
    return BinaryPolymerNetwork(molecules=molecules, food=food, reactions=reactions,
                                catalysts=catalysts, p=p, max_len=max_len,
                                food_len=food_len)
