"""RAF closure and exploitability.

`max_raf` is the standard Hordijk & Steel (2004) maximal-RAF algorithm. A reaction
set R' is a RAF over food set F when every reaction in R' is catalysed by a molecule
producible from F using R' (reflexively autocatalytic), and every reactant is itself
producible from F using R' (F-generated). The maximal RAF is unique and is reached by
iteratively discarding reactions that fail either condition.

`exploitability` measures how much of what a RAF produces contributes no catalysis
back to it. A molecule counts as an exploiter when the RAF produces it and it
catalyses nothing in the RAF -- reproduced by the closed set's catalysts, giving
nothing to closure in return. It is a cheap, purely structural observable; it says
nothing on its own about whether such a molecule could invade dynamically.

Several definitions are defensible, so one is fixed here as the PRIMARY and two
variants are computed alongside it, reported but never substituted for it:

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

from rafkit.binary_polymer import BinaryPolymerNetwork
from rafkit.catalysis import (catalysing_molecules, is_catalysed,
                              requires_non_food)


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
    """Molecules producible from the food set using `reactions`.

    Direction-aware: a ligation fires when both its reactants are present, a cleavage
    when its single reactant is. Cleavage **does** enlarge this set, because a polymer
    has many splits and need not be cleaved along the one it was built from -- see the
    `binary_polymer` docstring, where the contrary argument is recorded as refuted.
    """
    have = set(net.food)
    pending = [(net.reactants(r), net.products(r)) for r in reactions]
    changed = True
    while changed:
        changed = False
        still = []
        for reactants, products in pending:
            if all(x in have for x in reactants):
                for x in products:
                    if x not in have:
                        have.add(x)
                        changed = True
            else:
                still.append((reactants, products))
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
            if all(x in have for x in net.reactants(r))
            and is_catalysed(net.catalysts[r], have)
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
        reactants_in_raf.update(net.reactants(r))
        catalyses_in_raf |= (catalysing_molecules(net.catalysts[r]) & products)

    strict = products - catalyses_in_raf
    unused = strict - reactants_in_raf

    # `dispensable`: removing the molecule must not shrink the RAF. Removing m kills
    # every reaction producing or consuming m, and every reaction m alone catalysed.
    dispensable = set()
    for m in strict:
        survives = frozenset(
            r for r in raf.reactions
            if m not in net.reactions[r]
            and is_catalysed(net.catalysts[r], raf.closure - {m})
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


def _refine(net: BinaryPolymerNetwork, reactions: frozenset[int],
            strict: bool = False) -> frozenset[int]:
    """One RAF fixpoint restricted to a reaction subset.

    `strict=True` requires every catalyst to be a **non-food** molecule the set
    produces, which is the self-referential reading of "reflexively autocatalytic"
    (see `is_food_catalysed`). Default `False` is the literal Hordijk & Steel
    condition and is what `max_raf` and every prior result use.
    """
    current = reactions
    while True:
        have = _closure(net, current)
        keep = frozenset(
            r for r in current
            if all(x in have for x in net.reactants(r))
            and (requires_non_food(net.catalysts[r], have, net.food) if strict
                 else is_catalysed(net.catalysts[r], have))
        )
        if keep == current:
            return current
        current = keep
        if not current:
            return current


def max_raf_strict(net: BinaryPolymerNetwork) -> RafResult:
    """The maximal self-referential RAF: catalysts must be non-food products.

    The subset of the maximal RAF that actually needs its own output to run. This is
    the object a propagule would have to carry, so it -- not `max_raf` -- is where a
    count of lineages has to be taken.
    """
    current = _refine(net, frozenset(range(net.n_reactions)), strict=True)
    return RafResult(reactions=current, closure=_closure(net, current), n_rounds=0)


def sample_irrraf(net: BinaryPolymerNetwork, reactions: frozenset[int],
                  rng, strict: bool = False) -> frozenset[int]:
    """One irreducible RAF contained in `reactions`, by randomized shrinking.

    An irreducible RAF (Hordijk & Steel) is a RAF with no proper subset that is
    itself a RAF: a minimal self-sustaining core. It is the natural formal stand-in
    for a *lineage* -- the smallest thing a propagule has to carry to re-establish
    the network from food alone.

    Walk the reactions in a random order and try to drop each one, keeping the
    refined remainder whenever it is non-empty. One pass suffices, because maximal
    RAF is monotone in the reaction set: if dropping `r` collapses the set, it also
    collapses every subset, so a reaction that survives its own visit can never
    become removable later. Every reaction present at the end was therefore visited
    while present and found irremovable, which is the definition.

    The random order is what makes this a *sampler* -- different orders land in
    different irreducible cores. Distinct results are a lower bound on how many
    exist, never an upper one.

    **Prior art.** This is Steel, Hordijk & Smith, "Minimal autocatalytic networks"
    (arXiv:1212.4450, 2012), which describes the same remove-and-refine procedure and
    the same randomised re-ordering to sample. It was reinvented here on 2026-08-15
    and the attribution added on discovery. The same paper proves there may be exponentially many irrRAFs and that finding the
    smallest RAF is NP-hard, so a distinct-count that never saturates is the expected
    result rather than a surprising one.
    """
    current = _refine(net, reactions, strict=strict)
    order = list(current)
    rng.shuffle(order)
    for r in order:
        if r not in current:
            continue
        trial = _refine(net, current - {r}, strict=strict)
        if trial:
            current = trial
    return current


def irrraf_census(net: BinaryPolymerNetwork, raf: RafResult, n_samples: int,
                  rng, strict: bool = False) -> dict:
    """Sample irreducible RAFs and report how many distinct ones turn up.

    The count is the quantity of interest: it upper-bounds the number of
    distinguishable lineages the chemistry can carry, so a census of 1 means there
    is nothing to inherit and no ecology is possible regardless of the dynamics
    later placed on top.
    """
    if raf.is_empty:
        return {"n_samples": 0, "n_distinct": 0, "sizes": [], "mean_size": float("nan"),
                "mean_jaccard": float("nan"), "min_jaccard": float("nan"),
                "union_size": 0, "core_size": 0}

    found: list[frozenset[int]] = []
    seen: set[frozenset[int]] = set()
    for _ in range(n_samples):
        s = sample_irrraf(net, raf.reactions, rng, strict=strict)
        found.append(s)
        seen.add(s)

    distinct = sorted(seen, key=len)
    jac = []
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            a, b = distinct[i], distinct[j]
            jac.append(len(a & b) / len(a | b))
    self_ref = [c for c in distinct if not is_food_catalysed(net, c)]
    union: frozenset[int] = frozenset().union(*distinct)
    core: frozenset[int] = distinct[0]
    for s in distinct[1:]:
        core = core & s

    return {"n_samples": n_samples,
            "n_distinct": len(distinct),
            "n_self_referential": len(self_ref),
            "self_ref_sizes": [len(c) for c in self_ref],
            "sizes": [len(s) for s in distinct],
            "mean_size": sum(len(s) for s in found) / len(found),
            "mean_jaccard": sum(jac) / len(jac) if jac else float("nan"),
            "min_jaccard": min(jac) if jac else float("nan"),
            "union_size": len(union),
            "core_size": len(core)}


def is_food_catalysed(net: BinaryPolymerNetwork, core: frozenset[int]) -> bool:
    """Whether every reaction in `core` has a catalyst in the food set.

    Such a core is a RAF by the letter of the definition -- food is in the closure,
    so "catalysed by a molecule producible from F" is satisfied -- but it is not
    self-referential: it runs wherever the food runs, needs none of its own products,
    and therefore carries no heredity. A propagule is not required to establish it.

    This is a real degeneracy of the RAF definition rather than a quirk of E4, and it
    has to be split out before any count of cores can be read as a count of lineages.
    """
    if not core:
        return False
    return all(is_catalysed(net.catalysts[r], net.food) for r in core)


def catrenet_strictly_autocatalytic(net, raf: RafResult | None = None) -> frozenset[int]:
    """CatReNet's `strictlyAutocatalyticMaxRaf`, for cross-checking.

    CatReNet documents this as "a Max RAF that has the additional property that any
    contained reaction requires at least one molecule type for catalyzation that is
    not in the food set". Reproduced here by **black-box behavioural inference** from
    its published output on a generated network -- no CatReNet source was read, and
    none could be used, since it is GPL v3 and this library is MIT.

    The operation is a **filter on the maximal RAF, without re-refinement**: keep
    every reaction having at least one non-food catalyst, and stop. That is *not* the
    same as `max_raf_strict`, which imposes the same condition inside the fixpoint
    and therefore returns a set that is itself a RAF. Dropping reactions can break
    F-generation for the ones that remain, so this result need not be a RAF -- which
    is exactly why it is offered for interoperability rather than for analysis.

    On the committed CatReNet fixture: `max_raf` 183, this 175, `max_raf_strict` 161.
    """
    if raf is None:
        raf = max_raf(net)
    if raf.is_empty:
        return frozenset()
    return frozenset(r for r in raf.reactions
                     if requires_non_food(net.catalysts[r], raf.closure, net.food))


def core_raf(net, reactions=None) -> frozenset[int]:
    """`Core(Q)`: the reactions whose removal collapses the whole set.

    Defined by Huson, Xavier & Steel (2024) as
    ``Core(Q) = {r in R : phi(R \\ {r}) = empty}``. Their result: **this set is an RAF
    if and only if the system has a unique irreducible RAF**, and when it is, it *is*
    that iRAF. `has_unique_irraf` is the usable form of that test.
    """
    allowed = frozenset(range(net.n_reactions) if reactions is None else reactions)
    return frozenset(r for r in allowed if not _refine(net, allowed - {r}))


def has_unique_irraf(net, reactions=None) -> bool:
    """Whether the system has exactly one irreducible RAF, in polynomial time.

    Deciding how *many* iRAFs there are is hard in general -- there may be
    exponentially many, and finding the smallest is NP-hard (Steel, Hordijk & Smith
    2012) -- but the *unique* case is cheap, which is the point of `core_raf`.

    Returns False when there is no RAF at all: no iRAF is not one iRAF.
    """
    allowed = frozenset(range(net.n_reactions) if reactions is None else reactions)
    maximal = _refine(net, allowed)
    if not maximal:
        return False
    core = core_raf(net, allowed)
    return bool(core) and _refine(net, core) == core
