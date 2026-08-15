"""The catalysis relation χ, and the one predicate everything else is built on.

Huson, Xavier & Steel (2024) treat catalysis as a relation between **sets** of molecules
and reactions: reaction ``r`` proceeds when *some* catalyst set ``U`` in ``chi(r)`` is
entirely available. That single structure expresses three things a flat set of catalysts
cannot:

===========================  ============================================================
``chi(r)``                   meaning
===========================  ============================================================
``{{a}, {b}}``               *a* **or** *b* -- the simple, disjunctive case
``{{a, d}, {e}}``            (*a* **and** *d*) **or** *e* -- conjunctive requirements
``{}`` (no sets at all)      **must** be catalysed, and nothing catalyses it: never in a RAF
``{frozenset()}``            **may proceed uncatalysed**; always satisfied
===========================  ============================================================

The last two are a genuine distinction rather than a technicality -- in their §2.4 system
it is what separates a reaction that can join an RAF from one that cannot -- and writing
catalysts as a flat set collapses them.

`is_catalysed` handles all four rows without special-casing, because ``any()`` over no
sets is False and the empty set is a subset of everything.
"""
from __future__ import annotations

from typing import Iterable

CatalystSets = frozenset[frozenset[int]]


def normalise(spec) -> CatalystSets:
    """Accept either the simple or the general form and return the general one.

    A flat iterable of molecule indices -- the form used everywhere before conjunctive
    catalysis existed, and still the right one to write by hand for simple systems --
    becomes one singleton set per molecule, which is exactly equivalent.
    """
    if spec is None:
        return frozenset()
    out = []
    for item in spec:
        if isinstance(item, (frozenset, set, tuple, list)):
            out.append(frozenset(int(x) for x in item))
        else:
            out.append(frozenset({int(item)}))
    return frozenset(out)


def is_catalysed(chi: CatalystSets, available: Iterable[int]) -> bool:
    """Whether some catalyst set of a reaction is fully present.

    Note the two edge cases fall out rather than being handled: with no catalyst sets
    `any()` is False, so the reaction can never run; with the empty set present,
    ``frozenset() <= available`` is True, so it always can.
    """
    avail = available if isinstance(available, (set, frozenset)) else frozenset(available)
    return any(U <= avail for U in chi)


def catalysing_molecules(chi: CatalystSets) -> frozenset[int]:
    """Every molecule that appears in any catalyst set.

    The right notion of "x catalyses r" when catalysis is conjunctive: x may be
    necessary without being sufficient, and it still counts as catalysing.
    """
    return frozenset().union(*chi) if chi else frozenset()


def requires_non_food(chi: CatalystSets, available: Iterable[int], food) -> bool:
    """Whether some *satisfiable* catalyst set is not contained in the food set.

    This is the strictly-autocatalytic condition of Huson, Xavier & Steel (2024) §3.1
    stated exactly: ``U subset-of cl(F)`` and ``U not-subset-of F``. Under simple
    catalysis it reduces to "has a catalyst that is a non-food product", which is what
    `max_raf_strict` meant before this module existed.
    """
    avail = available if isinstance(available, (set, frozenset)) else frozenset(available)
    return any(U <= avail and not U <= food for U in chi)
