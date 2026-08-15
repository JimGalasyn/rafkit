"""RAF closure and exploitability.

`max_raf` is the standard Hordijk & Steel (2004) maximal-RAF algorithm. A reaction
set R' is a RAF over food set F when every reaction in R' is catalysed by a molecule
producible from F using R' (reflexively autocatalytic), and every reactant is itself
producible from F using R' (F-generated). The maximal RAF is unique and is reached by
iteratively discarding reactions that fail either condition.

`exploitability` is the observable `DESIGN_abiogenesis.md` §6a adds, and it exists to
test that section's load-bearing premise: **that exploiting closure is at least as
generic as closure itself.** A molecule counts as an exploiter when the RAF produces
it and it catalyses nothing in the RAF -- reproduced by the closed set's catalysts,
contributing no catalysis back.

That definition is one of several defensible ones and §6a flags the convention as
needing to be fixed in advance, so it is fixed here as the PRIMARY and two variants
are computed alongside it, reported but never substituted for it:

* `strict`   -- PRIMARY, as above.
* `unused`   -- produced by the RAF, catalyses nothing in the RAF, **and** is not a
                reactant of any RAF reaction. A strictly smaller set: these molecules
                are dead weight in both roles.
* `dispensable` -- produced by the RAF and removable without shrinking the RAF. The
                most demanding, and the closest to the hypercycle sense of a parasite:
                the network's throughput builds it and would lose nothing by not.
"""
from __future__ import annotations

from dataclasses import dataclass

from morphospace.chemistry.binary_polymer import BinaryPolymerNetwork


@dataclass(frozen=True)
class RafResult:
    reactions: frozenset[int]        # indices of the maximal RAF
    closure: frozenset[int]          # molecules producible from F using the RAF
    n_rounds: int                    # fixpoint iterations taken

    @property
    def size(self) -> int:
        return len(self.reactions)

    @property
    def is_empty(self) -> bool:
        return not self.reactions


def _closure(net: BinaryPolymerNetwork, reactions: frozenset[int]) -> frozenset[int]:
    """Molecules producible from the food set using `reactions`."""
    have = set(net.food)
    pending = [net.reactions[r] for r in reactions]
    changed = True
    while changed:
        changed = False
        still = []
        for a, b, ab in pending:
            if a in have and b in have:
                if ab not in have:
                    have.add(ab)
                    changed = True
            else:
                still.append((a, b, ab))
        pending = still
    return frozenset(have)


def max_raf(net: BinaryPolymerNetwork) -> RafResult:
    """The maximal RAF, by iterative removal to a fixpoint (Hordijk & Steel 2004)."""
    current = frozenset(range(net.n_reactions))
    rounds = 0
    while True:
        rounds += 1
        have = _closure(net, current)
        keep = frozenset(
            r for r in current
            if net.reactions[r][0] in have and net.reactions[r][1] in have
            and (net.catalysts[r] & have)
        )
        if keep == current:
            return RafResult(reactions=current, closure=have, n_rounds=rounds)
        current = keep
        if not current:
            return RafResult(reactions=current, closure=_closure(net, current),
                             n_rounds=rounds)


def exploitability(net: BinaryPolymerNetwork, raf: RafResult) -> dict:
    """Exploiter fractions among the molecules the RAF produces.

    Denominator is RAF products excluding food: food is supplied from outside, so
    counting it would inflate the fraction with molecules the network never had to
    make. Returns fractions and raw counts; `nan` fractions when the RAF is empty,
    which is honest -- there is nothing to exploit.
    """
    products = frozenset(raf.closure) - frozenset(net.food)
    n = len(products)
    if raf.is_empty or n == 0:
        nan = float("nan")
        return {"n_products": n, "strict": nan, "unused": nan, "dispensable": nan,
                "n_strict": 0, "n_unused": 0, "n_dispensable": 0}

    catalyses_in_raf = set()
    reactants_in_raf = set()
    for r in raf.reactions:
        a, b, _ = net.reactions[r]
        reactants_in_raf.add(a)
        reactants_in_raf.add(b)
        catalyses_in_raf |= (net.catalysts[r] & products)

    strict = products - catalyses_in_raf
    unused = strict - reactants_in_raf

    # `dispensable`: removing the molecule must not shrink the RAF. Removing m kills
    # every reaction producing or consuming m, and every reaction m alone catalysed.
    dispensable = set()
    for m in strict:
        survives = frozenset(
            r for r in raf.reactions
            if m not in net.reactions[r] and net.catalysts[r] & (raf.closure - {m})
        )
        sub = RafResult(reactions=survives, closure=_closure(net, survives),
                        n_rounds=0)
        again = _refine(net, sub.reactions)
        if len(again) == len(raf.reactions) - _touching(net, raf, m):
            dispensable.add(m)

    return {"n_products": n,
            "strict": len(strict) / n, "n_strict": len(strict),
            "unused": len(unused) / n, "n_unused": len(unused),
            "dispensable": len(dispensable) / n, "n_dispensable": len(dispensable)}


def _touching(net: BinaryPolymerNetwork, raf: RafResult, m: int) -> int:
    """RAF reactions that directly involve `m` as reactant or product."""
    return sum(1 for r in raf.reactions if m in net.reactions[r])


def _refine(net: BinaryPolymerNetwork, reactions: frozenset[int]) -> frozenset[int]:
    """One RAF fixpoint restricted to a reaction subset."""
    current = reactions
    while True:
        have = _closure(net, current)
        keep = frozenset(
            r for r in current
            if net.reactions[r][0] in have and net.reactions[r][1] in have
            and (net.catalysts[r] & have)
        )
        if keep == current:
            return current
        current = keep
        if not current:
            return current
