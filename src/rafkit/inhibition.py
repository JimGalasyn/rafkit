"""Uninhibited RAFs, where a molecule can prevent a reaction.

Hordijk & Steel (2012), Part II. Inhibition is given as ``k`` pairs ``(X_i, R_i)``:
every molecule in ``X_i`` inhibits every reaction in ``R_i``. A set ``R'`` is an
**uninhibited RAF** (u-RAF) when

* **(u-1)** ``R'`` is an RAF, and
* **(u-2)** ``R' ∩ R_i != empty`` implies ``supp(R') ∩ X_i = empty``,

where ``supp(R')`` is every molecule appearing as a reactant or product in ``R'``.

**Inhibition breaks the structure the rest of this library rests on.** Adding a reaction
can now *disable* another, so the maximal-RAF operator is no longer monotone -- and
monotonicity is what gives a *unique* maximum (Huson, Xavier & Steel 2024, lemma 3.1).
There is therefore no "the" maximal u-RAF: `max_urafs` returns a **collection**, and
that difference is in the signature deliberately rather than in a footnote. Deciding
whether a u-RAF exists at all is NP-complete.

What rescues it is that the problem is fixed-parameter tractable in ``k``, by their
theorem 1: the maximal u-RAFs are exactly the non-empty sets ``s(R_J ∩ R^J)`` as ``J``
ranges over subsets of ``[k]``. So the cost is ``2^k`` calls to the ordinary maximal-RAF
algorithm, and **``k`` is a property of how inhibition is encoded, not of the
chemistry**. One class per inhibited reaction makes ``2^k`` hopeless immediately;
`classes_from_inhibitors` groups by inhibiting *molecule* instead, which is what the
paper means by considering "types" of molecules that inhibit "types" of reactions.
"""
from __future__ import annotations

from itertools import combinations

from rafkit.raf import _refine

Inhibition = tuple[tuple[frozenset[int], frozenset[int]], ...]


def support(net, reactions) -> frozenset[int]:
    """Every molecule that is a reactant or product of some reaction in the set.

    Catalysts are deliberately excluded: the paper's ``supp`` is over reactants and
    products only, and including catalysts would make (u-2) strictly harder to satisfy.
    """
    out: set[int] = set()
    for r in reactions:
        out.update(net.reactants(r))
        out.update(net.products(r))
    return frozenset(out)


def classes_from_inhibitors(net) -> Inhibition:
    """Build the ``(X_i, R_i)`` classes from a network's per-reaction inhibitors.

    Grouped by inhibiting **molecule**, so ``k`` is the number of distinct inhibitors
    rather than the number of inhibited reactions. That choice is the difference
    between a feasible ``2^k`` and an impossible one, and it costs nothing: the pair
    ``({x}, {reactions x inhibits})`` expresses exactly the same relation.
    """
    by_molecule: dict[int, set[int]] = {}
    for r, inhibitors in enumerate(getattr(net, "inhibitors", ()) or ()):
        for x in inhibitors:
            by_molecule.setdefault(x, set()).add(r)
    return tuple((frozenset({x}), frozenset(rs))
                 for x, rs in sorted(by_molecule.items()))


def is_uninhibited(net, reactions, inhibition: Inhibition) -> bool:
    """Condition (u-2): nothing the set makes or uses inhibits anything the set does."""
    rs = frozenset(reactions)
    supp = support(net, rs)
    return all(not (rs & R_i) or not (supp & X_i) for X_i, R_i in inhibition)


def is_uraf(net, reactions, inhibition: Inhibition) -> bool:
    """Both conditions: an RAF that inhibits none of its own reactions."""
    rs = frozenset(reactions)
    return bool(rs) and _refine(net, rs) == rs and is_uninhibited(net, rs, inhibition)


def max_urafs(net, inhibition: Inhibition | None = None,
              reactions=None) -> tuple[frozenset[int], ...]:
    """All maximal uninhibited RAFs, by Hordijk & Steel (2012) theorem 1.

    Returns a tuple because there is generally more than one and none of them is
    canonical -- see the module docstring. Empty tuple means no u-RAF exists.

    Cost is ``2^k`` maximal-RAF computations, with ``k = len(inhibition)``.
    """
    if inhibition is None:
        inhibition = classes_from_inhibitors(net)
    allowed = frozenset(range(net.n_reactions) if reactions is None else reactions)
    if not inhibition:
        maximal = _refine(net, allowed)
        return (maximal,) if maximal else ()

    k = len(inhibition)
    found: set[frozenset[int]] = set()
    for size in range(k + 1):
        for J in combinations(range(k), size):
            Jset = set(J)
            # R_J: reactions untouched by every class NOT in J.
            R_J = frozenset(
                r for r in allowed
                if all(not (support(net, {r}) & inhibition[j][0])
                       for j in range(k) if j not in Jset))
            # R^J: reactions inhibited by no class IN J.
            R_super = frozenset(
                r for r in allowed if all(r not in inhibition[j][1] for j in Jset))
            candidate = _refine(net, R_J & R_super)
            if candidate:
                found.add(candidate)

    # Different J can yield nested results; only the maximal ones are u-RAFs by (iii).
    return tuple(sorted((s for s in found
                         if not any(s < t for t in found)),
                        key=lambda s: (-len(s), sorted(s))))
