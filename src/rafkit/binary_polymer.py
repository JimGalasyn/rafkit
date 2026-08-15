"""E4 — Kauffman's binary polymer model.

The canonical setting for the closure phase transition (Kauffman 1986; Hordijk &
Steel 2004). It is the substrate `DESIGN_abiogenesis.md` §6a's premise 1 was
established in, which is why premise 2 must be measured here and not on E1:
ReactionAtlas carries reactions with DFT barriers but **no catalysis relation**, and
exploitability is defined through catalysis.

Molecules are binary strings of length 1..`max_len`. The food set is every string of
length <= `food_len`. Reactions are **ligations** `a + b -> ab` for every ordered pair
whose concatenation is within `max_len`, and -- when `cleavage=True` -- the reverse
`ab -> a + b` as well.

Cleavage was originally omitted on the grounds that it "makes every product trivially
reachable". That is overstated but **directionally right**, and a first attempt to
overturn it here was wrong and is recorded so it is not retried: the argument was that
a cleavage can only fire on a molecule you already have, and a long molecule was
ligated from shorter ones that must therefore already be present, so the closure could
not grow.

**Measured, it grows a lot** -- at `max_len=6`, `p=0.004`, from 6-15 molecules to ~120
in three of four seeds. The flaw in the argument is that a polymer has *many* splits:
`0101` may be reachable only as `01`+`01`, while the cleavage `0101 -> 010 + 1` yields
`010`, which no catalysed ligation produces. Production and cleavage need not use the
same split, so cleavage genuinely enlarges reachability.

That makes cleavage the leading candidate for the transition discrepancy of
`DESIGN_abiogenesis.md` §6b (published f ~ 1.20 against ~3 here): more reachable
molecules means more catalysts in the closure, at any catalysis level.

It matters because the two published binary-polymer references both use
cleavage-ligation chemistries: Steel, Hordijk & Smith (2012), against whose f ~ 1.20
transition this model is uncalibrated, and Serra & Villani (2026), whose C-BPM is the
comparison ensemble. See `DESIGN_abiogenesis.md` §6b and §6c.

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

    `reactions[i] = (a, b, ab)` as molecule indices, always stored in that order
    regardless of which way the reaction runs. `directions[i]` is `+1` for a ligation
    `a + b -> ab` and `-1` for a cleavage `ab -> a + b`; use `reactants()` and
    `products()` rather than unpacking the triple by hand. `catalysts[i]` is the set of
    molecule indices catalysing reaction `i`.
    """

    molecules: tuple[str, ...]
    food: frozenset[int]
    reactions: tuple[tuple[int, int, int], ...]
    catalysts: tuple[frozenset[int], ...]
    p: float
    max_len: int
    food_len: int
    directions: tuple[int, ...] = ()

    def __post_init__(self):
        if not self.directions:                      # default: all ligations
            object.__setattr__(self, "directions", (1,) * len(self.reactions))
        elif len(self.directions) != len(self.reactions):
            raise ValueError(
                f"directions has {len(self.directions)} entries for "
                f"{len(self.reactions)} reactions")

    def reactants(self, r: int) -> tuple[int, ...]:
        """Molecules consumed by reaction `r`, in its stored direction."""
        a, b, ab = self.reactions[r]
        return (a, b) if self.directions[r] > 0 else (ab,)

    def products(self, r: int) -> tuple[int, ...]:
        """Molecules produced by reaction `r`, in its stored direction."""
        a, b, ab = self.reactions[r]
        return (ab,) if self.directions[r] > 0 else (a, b)

    @property
    def n_cleavages(self) -> int:
        return sum(1 for d in self.directions if d < 0)

    @property
    def catalysis_level(self) -> float:
        """`f` in the published convention: catalysed reactions per molecule.

        Steel, Hordijk & Smith (2012) define f = p|R| where **R counts reversible
        `cleavage-ligation` reactions**, i.e. a ligation and its reverse are ONE
        reaction. Reporting `mean_catalysed_per_molecule` against their f therefore
        double-counts a cleavage chemistry and lands 2x too high. This property is the
        comparable quantity; use it, and not `mean_catalysed_per_molecule`, whenever a
        number is set beside theirs (`DESIGN_abiogenesis.md` §6b).
        """
        if not self.molecules:
            return 0.0
        n_pairs = self.n_reactions - self.n_cleavages
        return sum(len(c) for c in self.catalysts[:n_pairs]) / len(self.molecules)

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
                   rng: np.random.Generator | None = None,
                   cleavage: bool = False,
                   paired_catalysis: bool = True) -> BinaryPolymerNetwork:
    """Generate one binary-polymer network with catalysis at probability `p`.

    `cleavage=True` adds the reverse `ab -> a + b` of every ligation, doubling the
    stored reaction count.

    `paired_catalysis` (default True) makes a ligation and its reverse **one
    catalysed unit**, sharing a catalyst set -- the reversible "cleavage-ligation"
    reaction of Steel, Hordijk & Smith (2012), and the convention their f = p|R| is
    measured in. Set False to draw the two directions independently; that is a
    different chemistry and its f is not comparable to theirs.
    """
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

    ligations = tuple(
        (index[a], index[b], index[a + b])
        for a in molecules for b in molecules if len(a) + len(b) <= max_len
    )
    if cleavage:
        reactions = ligations + ligations
        directions = (1,) * len(ligations) + (-1,) * len(ligations)
    else:
        reactions = ligations
        directions = (1,) * len(ligations)

    # Sparse Bernoulli draw: sampling the number of catalysts per reaction and then
    # which ones is O(edges) rather than O(molecules x reactions), which matters --
    # the dense product is ~10^8 at max_len=10.
    n_mol = len(molecules)
    n_draw = len(ligations) if (cleavage and paired_catalysis) else len(reactions)
    counts = rng.binomial(n_mol, p, size=n_draw)
    drawn = tuple(
        frozenset(rng.choice(n_mol, size=int(k), replace=False).tolist()) if k else
        frozenset()
        for k in counts
    )
    # Paired: the cleavage half re-uses its ligation's catalysts rather than redrawing.
    catalysts = drawn + drawn if (cleavage and paired_catalysis) else drawn
    return BinaryPolymerNetwork(molecules=molecules, food=food, reactions=reactions,
                                catalysts=catalysts, p=p, max_len=max_len,
                                food_len=food_len, directions=directions)
