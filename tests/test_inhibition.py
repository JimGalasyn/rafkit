"""Uninhibited RAFs — Hordijk & Steel (2012), Part II.

The load-bearing test here is `test_matches_brute_force`: for networks small enough to
enumerate every subset, the fixed-parameter algorithm is checked against a direct
search that shares no code with it. Everything else in this file is a property from the
paper, asserted rather than assumed.
"""
from __future__ import annotations

from itertools import chain, combinations

import numpy as np
import pytest

from rafkit import binary_polymer, max_raf, parse_crs
from rafkit.inhibition import (classes_from_inhibitors, is_uninhibited, is_uraf,
                               max_urafs, support)
from rafkit.raf import _refine


def _powerset(items):
    items = list(items)
    return chain.from_iterable(combinations(items, k) for k in range(len(items) + 1))


def _brute_force_max_urafs(net, inhibition):
    """Every maximal u-RAF, by direct enumeration. Shares no code with max_urafs."""
    urafs = [frozenset(s) for s in _powerset(range(net.n_reactions))
             if s and is_uraf(net, s, inhibition)]
    return {s for s in urafs if not any(s < t for t in urafs)}


class TestDefinition:
    def test_support_is_reactants_and_products_only(self):
        net = parse_crs("Food: a, b\nr1 : a + b [z] => c\n")
        names = {net.molecules[m] for m in support(net, {0})}
        assert names == {"a", "b", "c"}       # z is a catalyst, not in the support

    def test_inhibition_by_something_absent_from_the_support_is_harmless(self):
        # z inhibits r1, but z is neither reactant nor product of r1, so (u-2) holds.
        net = parse_crs("Food: a, b\nr1 : a + b [a] {z} => c\n")
        assert is_uninhibited(net, {0}, classes_from_inhibitors(net))
        assert is_uraf(net, {0}, classes_from_inhibitors(net))

    def test_a_set_that_inhibits_its_own_reaction_is_not_a_uraf(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        c = net.molecules.index("c")
        inh = ((frozenset({c}), frozenset({0})),)
        assert _refine(net, {0}) == {0}            # it IS an RAF
        assert not is_uraf(net, {0}, inh)          # but not uninhibited

    def test_raf_is_unaffected_by_inhibition(self):
        """The paper defines RAF without reference to inhibition; (u-2) is separate.
        CatReNet instead filters inhibited reactions inside its maxRaf, so its result
        differs from the definition here whenever inhibitors are present."""
        net = parse_crs("Food: a, b\nr1 : a + b [a] => c\nr2 : a + c [a] => d\n")
        assert len(max_raf(net).reactions) == 2


class TestTheorem1:
    def test_a_subset_of_a_uraf_that_is_a_raf_is_a_uraf(self):
        """Stated in the paper: (u-2) is inherited downward."""
        net = parse_crs(
            "Food: a, b\nr1 : a + b [a] => c\nr2 : a + c [a] => d\nr3 : a + d [a] => e\n")
        e = net.molecules.index("e")
        inh = ((frozenset({e}), frozenset({2})),)
        whole = max_urafs(net, inh)
        assert whole
        for u in whole:
            for s in _powerset(u):
                s = frozenset(s)
                if s and _refine(net, s) == s:
                    assert is_uraf(net, s, inh)

    # Seeds probed to give a maximal RAF of 4-12 reactions with >=2 produced
    # molecules, so exhaustive enumeration is feasible and the constraint bites.
    @pytest.mark.parametrize("seed", [1, 5, 7, 12, 13, 15, 16])
    def test_matches_brute_force(self, seed):
        """The fixed-parameter algorithm against direct enumeration of every subset.

        The two computations share no code: one walks subsets of [k] and calls the
        maximal-RAF fixpoint, the other tests every subset of the RAF directly.
        """
        net = binary_polymer(max_len=4, food_len=2, p=0.01,
                             rng=np.random.default_rng(seed), cleavage=True)
        raf = sorted(max_raf(net).reactions)
        assert 3 <= len(raf) <= 12, "seed no longer yields an enumerable RAF"

        # Inhibit reactions by molecules the set actually makes, so (u-2) can fail.
        produced = sorted(support(net, raf) - net.food)
        assert len(produced) >= 2
        inh = tuple((frozenset({produced[i]}), frozenset({raf[i]}))
                    for i in range(2))

        assert set(max_urafs(net, inh, reactions=raf)) == \
            _brute_force_max_urafs_on(net, raf, inh)


def _brute_force_max_urafs_on(net, allowed, inhibition):
    urafs = [frozenset(s) for s in _powerset(allowed)
             if s and is_uraf(net, s, inhibition)]
    return {s for s in urafs if not any(s < t for t in urafs)}


class TestCollectionSemantics:
    def test_there_can_be_more_than_one_maximal_uraf(self):
        """The structural break: no unique maximum, so the API returns a collection.

        Two independent always-on reactions, each inhibited by the other's product.
        Neither can coexist with the other, and neither is canonical.
        """
        net = parse_crs(
            "Food: a, b\nr1 : a + b [a] => c\nr2 : a + b [b] => d\n")
        c, d = net.molecules.index("c"), net.molecules.index("d")
        inh = ((frozenset({c}), frozenset({1})), (frozenset({d}), frozenset({0})))
        got = {frozenset(net.names[r] for r in u) for u in max_urafs(net, inh)}
        assert got == {frozenset({"r1"}), frozenset({"r2"})}

    def test_no_inhibition_gives_back_the_maximal_raf(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        assert max_urafs(net, ()) == (max_raf(net).reactions,)

    def test_no_uraf_returns_an_empty_collection(self):
        net = parse_crs("Food: a, b\nr1 : a + b [c] => c\n")
        c = net.molecules.index("c")
        assert max_urafs(net, ((frozenset({c}), frozenset({0})),)) == ()


class TestClassEncoding:
    def test_grouping_is_by_inhibiting_molecule_not_by_reaction(self):
        """k is the cost, and it is a property of the encoding. Grouping by molecule
        keeps k at the number of distinct inhibitors rather than of inhibited
        reactions, which is the difference between 2^k feasible and not."""
        net = parse_crs(
            "Food: a, b\nr1 : a + b [a] {z} => c\nr2 : a + b [b] {z} => d\n")
        classes = classes_from_inhibitors(net)
        assert len(classes) == 1                       # one inhibitor, not two reactions
        (X, R), = classes
        assert {net.molecules[x] for x in X} == {"z"}
        assert R == frozenset({0, 1})


class TestDivergenceFromCatReNet:
    """CatReNet computes something different once inhibitors are present.

    Its `maxRaf` filters inhibited reactions *during* the RAF computation, where
    Hordijk & Steel (2012) define an RAF without reference to inhibition and add (u-2)
    as a separate condition on u-RAFs. On the network below CatReNet's `maxRaf` and
    `uRaf` both return nothing, while two maximal u-RAFs exist by the definition.

    Verified by hand rather than asserted: `{r1}` is an RAF, its support is {a,b,c},
    and the only class inhibiting r1 is {d} — so (u-2) holds. Likewise `{r2}`. Their
    union is an RAF but fails (u-2), which is why neither can be extended.
    """

    NET = "Food: a, b\nr1 : a + b [a] {d} => c\nr2 : a + b [b] {c} => d\n"

    def test_two_maximal_urafs_exist_where_catrenet_reports_none(self):
        net = parse_crs(self.NET)
        got = {frozenset(net.names[r] for r in u) for u in max_urafs(net)}
        assert got == {frozenset({"r1"}), frozenset({"r2"})}

    def test_each_is_a_raf_satisfying_u2_but_their_union_is_not(self):
        net = parse_crs(self.NET)
        cls = classes_from_inhibitors(net)
        assert is_uraf(net, {0}, cls) and is_uraf(net, {1}, cls)
        assert _refine(net, {0, 1}) == {0, 1}          # the union IS an RAF
        assert not is_uninhibited(net, {0, 1}, cls)    # but violates (u-2)
