"""A general catalytic reaction network.

`BinaryPolymerNetwork` stores its reactions compactly as `(a, b, ab)` triples, which
is what makes tens of thousands of them cheap, but it can only express ligation and
cleavage over binary strings. `ReactionNetwork` expresses an arbitrary catalytic
reaction system -- any molecule names, any number of reactants and products -- and
exposes the **same small protocol** the RAF algorithms use, so `max_raf`,
`max_raf_strict`, `sample_irrraf`, `irrraf_census` and `exploitability` all work on
either without change:

    food                molecule indices supplied from outside
    molecules           names, indexed
    n_molecules
    n_reactions
    catalysts[r]        frozenset of molecule indices catalysing reaction r
    reactants(r)        molecules consumed by r
    products(r)         molecules produced by r
    reactions[r]        every molecule involved in r (membership tests only)

This is what makes the library usable on networks that did not come from a polymer
model -- see `rafkit.crs` for reading them from CatReNet's CRS format.
"""
from __future__ import annotations

from dataclasses import dataclass

from rafkit.catalysis import normalise


@dataclass(frozen=True)
class ReactionNetwork:
    """An arbitrary catalytic reaction system.

    `reaction_pairs[i] = (reactants, products)` as tuples of molecule indices.
    Reactants and products are treated as **sets**: a reaction `X + X -> Y` requires
    only that `X` be present, so listing `X` once is not a loss of information for
    any RAF computation (it would be for stoichiometry, which this class does not
    model).
    """

    molecules: tuple[str, ...]
    food: frozenset[int]
    reaction_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    catalysts: tuple[frozenset[int], ...]
    names: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "catalysts",
                           tuple(normalise(c) for c in self.catalysts))
        if len(self.catalysts) != len(self.reaction_pairs):
            raise ValueError(
                f"{len(self.catalysts)} catalyst sets for "
                f"{len(self.reaction_pairs)} reactions")
        if not self.names:
            object.__setattr__(
                self, "names", tuple(f"r{i + 1}" for i in range(len(self.reaction_pairs))))
        elif len(self.names) != len(self.reaction_pairs):
            raise ValueError(
                f"{len(self.names)} names for {len(self.reaction_pairs)} reactions")

    @property
    def n_molecules(self) -> int:
        return len(self.molecules)

    @property
    def n_reactions(self) -> int:
        return len(self.reaction_pairs)

    @property
    def reactions(self) -> tuple[tuple[int, ...], ...]:
        """Every molecule involved in each reaction; for membership tests only."""
        return tuple(r + p for r, p in self.reaction_pairs)

    def reactants(self, r: int) -> tuple[int, ...]:
        return self.reaction_pairs[r][0]

    def products(self, r: int) -> tuple[int, ...]:
        return self.reaction_pairs[r][1]

    @property
    def mean_catalysed_per_molecule(self) -> float:
        if not self.molecules:
            return 0.0
        return sum(len(c) for c in self.catalysts) / len(self.molecules)

    catalysis_level = mean_catalysed_per_molecule
