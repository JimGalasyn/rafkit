"""Hand-computable correctness tests for the stochastic simulator.

A Gillespie implementation that is subtly wrong produces plausible trajectories rather
than errors, so these check arithmetic that can be worked out on paper, plus the one
structural invariant that caught two real defects while this module was being written:
**a molecule can never appear before its reactants.**
"""
from __future__ import annotations

import numpy as np
import pytest
from rafkit import binary_polymer, max_raf
from rafkit.network import ReactionNetwork

from rafkit.gillespie import (UNCATALYSED_FACTOR, _pair_count, propensities,
                                   simulate)


def _net(molecules, food, pairs, catalysts):
    return ReactionNetwork(molecules=tuple(molecules), food=frozenset(food),
                           reaction_pairs=tuple(pairs),
                           catalysts=tuple(frozenset(c) for c in catalysts))


class TestPairCount:
    def test_distinct_reactants_multiply(self):
        assert _pair_count(np.array([3, 4, 0]), (0, 1)) == 12

    def test_a_species_cannot_react_with_itself(self):
        # 5 copies give 10 distinct pairs, not 25: n(n-1)/2.
        assert _pair_count(np.array([5, 0, 0]), (0, 0)) == 10

    def test_one_copy_gives_no_self_pair(self):
        assert _pair_count(np.array([1, 0, 0]), (0, 0)) == 0

    def test_single_reactant_is_its_own_count(self):
        assert _pair_count(np.array([0, 0, 7]), (2,)) == 7


class TestPropensities:
    def test_catalysed_runs_at_full_rate(self):
        n = _net("abc", [0, 1], [((0, 1), (2,))], [{2}])
        counts = np.array([3, 4, 1])          # catalyst c present
        assert propensities(n, counts)[0] == pytest.approx(12.0)

    def test_uncatalysed_runs_at_the_reduced_rate(self):
        n = _net("abc", [0, 1], [((0, 1), (2,))], [{2}])
        counts = np.array([3, 4, 0])          # catalyst absent
        assert propensities(n, counts)[0] == pytest.approx(12.0 / UNCATALYSED_FACTOR)

    def test_the_reduction_factor_is_exactly_the_ratio(self):
        n = _net("abc", [0, 1], [((0, 1), (2,))], [{2}])
        with_cat = propensities(n, np.array([3, 4, 1]))[0]
        without = propensities(n, np.array([3, 4, 0]))[0]
        assert with_cat / without == pytest.approx(UNCATALYSED_FACTOR)

    def test_missing_reactant_gives_zero(self):
        n = _net("abc", [0], [((0, 1), (2,))], [{2}])
        assert propensities(n, np.array([9, 0, 9]))[0] == 0.0

    def test_restriction_silences_excluded_reactions(self):
        n = _net("abcd", [0, 1], [((0, 1), (2,)), ((0, 1), (3,))], [{2}, {3}])
        a = propensities(n, np.array([3, 4, 1, 1]), reactions=[0])
        assert a[0] > 0 and a[1] == 0.0


class TestSimulation:
    def test_food_is_never_depleted_below_the_floor(self):
        net = binary_polymer(max_len=4, food_len=2, p=0.02,
                             rng=np.random.default_rng(0), cleavage=True)
        tr = simulate(net, n_events=800, rng=np.random.default_rng(1), food_floor=5)
        for f in net.food:
            assert tr.counts[:, f].min() >= 5

    def test_same_seed_gives_the_same_trajectory(self):
        net = binary_polymer(max_len=4, food_len=2, p=0.02,
                             rng=np.random.default_rng(0), cleavage=True)
        kw = dict(n_events=500, reactions=sorted(max_raf(net).reactions))
        a = simulate(net, rng=np.random.default_rng(7), **kw)
        b = simulate(net, rng=np.random.default_rng(7), **kw)
        assert a.first_appearance == b.first_appearance
        assert np.array_equal(a.counts, b.counts)

    def test_every_molecule_has_a_route_that_predates_it(self):
        """The invariant that caught three defects while this was written.

        For each non-food molecule, **at least one** reaction producing it must have all
        its reactants present at or before it appeared. Requiring that of *every*
        producing reaction is the wrong claim -- a species is typically also a cleavage
        product of something larger and later -- and asserting it fails on a correct
        simulator, which is how the first version of this test behaved.
        """
        net = binary_polymer(max_len=5, food_len=2, p=0.0045,
                             rng=np.random.default_rng(540), cleavage=True)
        raf = sorted(max_raf(net).reactions)
        tr = simulate(net, n_events=4000, rng=np.random.default_rng(3), reactions=raf)

        for mol, t_mol in tr.first_appearance.items():
            if mol in net.food:
                continue                      # supplied from outside; no route needed
            routes = [r for r in raf if mol in net.products(r)]
            assert routes, f"{net.molecules[mol]} appeared but nothing produces it"
            assert any(
                all((t := tr.first_appearance.get(x)) is not None and t <= t_mol
                    for x in net.reactants(r))
                for r in routes
            ), f"{net.molecules[mol]} appeared before any route to it was available"

    def test_a_dead_network_stops_rather_than_spinning(self):
        n = _net("ab", [0], [((0, 1), (1,))], [{1}])   # reactant b never exists
        tr = simulate(n, n_events=100, rng=np.random.default_rng(0))
        assert tr.times[-1] == 0.0
