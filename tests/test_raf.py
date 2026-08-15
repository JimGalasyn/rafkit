"""RAF closure, exploitability, and the E4 generator.

The load-bearing tests are the hand-computed networks in `TestMaxRaf`: RAF is the
predictor everything in `DESIGN_abiogenesis.md` §6 rests on, and a RAF algorithm that
is subtly wrong produces plausible numbers rather than errors.
"""
from __future__ import annotations

import numpy as np
import pytest

from morphospace.chemistry.binary_polymer import BinaryPolymerNetwork, binary_polymer
from morphospace.chemistry.raf import (
    _refine, exploitability, irrraf_census, is_food_catalysed, max_raf,
    max_raf_strict, sample_irrraf)


def _net(molecules, food, reactions, catalysts):
    """Hand-built network; the generator is not involved."""
    return BinaryPolymerNetwork(
        molecules=tuple(molecules), food=frozenset(food),
        reactions=tuple(reactions), catalysts=tuple(frozenset(c) for c in catalysts),
        p=0.0, max_len=0, food_len=0)


class TestMaxRaf:
    def test_no_catalysis_gives_an_empty_raf(self):
        """Reflexive autocatalysis is the whole point; uncatalysed reactions cannot
        be in a RAF however well supplied their reactants are."""
        n = _net("ab|c", food=[0, 1], reactions=[(0, 1, 2)], catalysts=[set()])
        assert max_raf(n).is_empty

    def test_a_reaction_catalysed_by_food_survives(self):
        n = _net("ab|c", food=[0, 1], reactions=[(0, 1, 2)], catalysts=[{0}])
        r = max_raf(n)
        assert r.reactions == frozenset({0})
        assert r.closure == frozenset({0, 1, 2})

    def test_a_reaction_catalysed_by_its_OWN_product_survives(self):
        """The canonical autocatalytic case, and the one a naive implementation
        drops: the catalyst does not exist until the reaction runs, but it is
        producible from food *using this reaction*, which is what RAF requires."""
        n = _net("ab|c", food=[0, 1], reactions=[(0, 1, 2)], catalysts=[{2}])
        assert max_raf(n).reactions == frozenset({0})

    def test_a_reaction_whose_only_catalyst_is_unreachable_is_removed(self):
        n = _net("ab|c|d", food=[0, 1], reactions=[(0, 1, 2)], catalysts=[{3}])
        assert max_raf(n).is_empty

    def test_a_reaction_whose_reactant_is_unreachable_is_removed(self):
        # r0 needs molecule 3, which nothing produces and food does not supply.
        n = _net("ab|c|d", food=[0, 1], reactions=[(0, 3, 2)], catalysts=[{0}])
        assert max_raf(n).is_empty

    def test_removal_cascades(self):
        """r1's reactant is r0's product, and r0's catalyst is unreachable. Killing
        r0 must kill r1 -- one round of removal is not enough."""
        mols = "abcde"
        n = _net(mols, food=[0, 1],
                 reactions=[(0, 1, 2), (2, 0, 3)],
                 catalysts=[{4}, {0}])
        r = max_raf(n)
        assert r.is_empty
        assert r.n_rounds >= 2, "cascade needs more than one fixpoint round"

    def test_a_two_reaction_raf_is_kept_whole(self):
        mols = "abcde"
        n = _net(mols, food=[0, 1],
                 reactions=[(0, 1, 2), (2, 0, 3)],
                 catalysts=[{0}, {2}])
        assert max_raf(n).reactions == frozenset({0, 1})


class TestExploitability:
    def test_a_product_that_catalyses_nothing_is_an_exploiter(self):
        # r0: a+b -> c catalysed by food a. c catalyses nothing => strict exploiter.
        n = _net("abc", food=[0, 1], reactions=[(0, 1, 2)], catalysts=[{0}])
        e = exploitability(n, max_raf(n))
        assert e["n_products"] == 1 and e["n_strict"] == 1 and e["strict"] == 1.0

    def test_a_product_that_catalyses_IS_NOT_an_exploiter(self):
        n = _net("abc", food=[0, 1], reactions=[(0, 1, 2)], catalysts=[{2}])
        e = exploitability(n, max_raf(n))
        assert e["n_products"] == 1 and e["n_strict"] == 0 and e["strict"] == 0.0

    def test_food_is_excluded_from_the_denominator(self):
        """Food is supplied from outside; counting it would inflate the fraction
        with molecules the network never had to make."""
        n = _net("abc", food=[0, 1], reactions=[(0, 1, 2)], catalysts=[{0}])
        assert exploitability(n, max_raf(n))["n_products"] == 1     # not 3

    def test_an_empty_raf_reports_nan_not_zero(self):
        """No RAF means nothing to exploit. Zero would read as 'measured, and none'."""
        n = _net("abc", food=[0, 1], reactions=[(0, 1, 2)], catalysts=[set()])
        e = exploitability(n, max_raf(n))
        assert np.isnan(e["strict"]) and e["n_products"] == 0

    def test_the_variants_are_nested(self):
        """dispensable <= unused <= strict, by construction. If this inverts, one of
        the three definitions has drifted from its docstring."""
        rng = np.random.default_rng(3)
        for p in (2e-3, 5e-3, 1e-2):
            n = binary_polymer(max_len=6, food_len=2, p=p, rng=rng)
            r = max_raf(n)
            if r.is_empty:
                continue
            e = exploitability(n, r)
            assert e["n_unused"] <= e["n_strict"]
            assert e["n_dispensable"] <= e["n_strict"]


class TestGenerator:
    def test_shape_and_food(self):
        n = binary_polymer(max_len=4, food_len=2, p=0.0)
        assert n.n_molecules == 2 + 4 + 8 + 16          # strings of length 1..4
        assert len(n.food) == 2 + 4                     # length 1..2
        assert all(len(n.molecules[a]) + len(n.molecules[b]) ==
                   len(n.molecules[ab]) for a, b, ab in n.reactions)

    def test_zero_p_gives_no_catalysis_and_no_raf(self):
        n = binary_polymer(max_len=5, food_len=2, p=0.0, rng=np.random.default_rng(0))
        assert n.mean_catalysed_per_molecule == 0.0
        assert max_raf(n).is_empty

    def test_p_one_catalyses_everything(self):
        n = binary_polymer(max_len=4, food_len=2, p=1.0, rng=np.random.default_rng(0))
        assert all(len(c) == n.n_molecules for c in n.catalysts)

    def test_seeded_generation_is_reproducible(self):
        a = binary_polymer(max_len=5, food_len=2, p=5e-3, rng=np.random.default_rng(7))
        b = binary_polymer(max_len=5, food_len=2, p=5e-3, rng=np.random.default_rng(7))
        assert a.catalysts == b.catalysts

    @pytest.mark.parametrize("bad", [{"max_len": 1}, {"food_len": 9}, {"p": 1.5}])
    def test_invalid_parameters_raise(self, bad):
        kw = {"max_len": 5, "food_len": 2, "p": 1e-3} | bad
        with pytest.raises(ValueError):
            binary_polymer(**kw)


class TestIrrRaf:
    """Hand-computed irreducible RAFs.

    Same reasoning as `TestMaxRaf`: a sampler that quietly returns something which is
    not irreducible, or misses cores that exist, produces a plausible census rather
    than an error -- and the census is the number the ecology argument rests on.
    """

    def test_two_disjoint_cores_are_both_found(self):
        # a+b -> c catalysed by c, and a+b -> d catalysed by d. Each is a RAF on its
        # own, they share only the food, so there are exactly two irreducible cores.
        n = _net("abcd", food=[0, 1],
                 reactions=[(0, 1, 2), (0, 1, 3)], catalysts=[{2}, {3}])
        raf = max_raf(n)
        assert raf.reactions == frozenset({0, 1})
        c = irrraf_census(n, raf, n_samples=20, rng=np.random.default_rng(0))
        assert c["n_distinct"] == 2
        assert c["sizes"] == [1, 1]
        assert c["mean_jaccard"] == 0.0      # disjoint
        assert c["core_size"] == 0           # no shared reaction
        assert c["union_size"] == 2

    def test_an_interdependent_pair_is_a_single_irreducible_core(self):
        # a+b -> c catalysed by d; a+c -> d catalysed by c. Neither reaction survives
        # without the other, so the maximal RAF is already irreducible.
        n = _net("abcd", food=[0, 1],
                 reactions=[(0, 1, 2), (0, 2, 3)], catalysts=[{3}, {2}])
        raf = max_raf(n)
        assert raf.reactions == frozenset({0, 1})
        c = irrraf_census(n, raf, n_samples=10, rng=np.random.default_rng(0))
        assert c["n_distinct"] == 1
        assert c["sizes"] == [2]
        assert c["core_size"] == 2

    def test_a_sampled_core_is_itself_a_raf(self):
        net = binary_polymer(max_len=6, food_len=2, p=0.01,
                             rng=np.random.default_rng(3))
        raf = max_raf(net)
        assert not raf.is_empty
        s = sample_irrraf(net, raf.reactions, np.random.default_rng(1))
        assert s, "sampler returned an empty set"
        assert _refine(net, s) == s, "sampled core is not a fixpoint"

    def test_a_sampled_core_is_irreducible(self):
        net = binary_polymer(max_len=6, food_len=2, p=0.01,
                             rng=np.random.default_rng(3))
        s = sample_irrraf(net, max_raf(net).reactions, np.random.default_rng(1))
        for r in s:
            assert not _refine(net, s - {r}), f"dropping {r} left a RAF: not minimal"

    def test_empty_raf_gives_an_empty_census(self):
        n = _net("abc", food=[0, 1], reactions=[(0, 1, 2)], catalysts=[set()])
        c = irrraf_census(n, max_raf(n), n_samples=5, rng=np.random.default_rng(0))
        assert c["n_distinct"] == 0


class TestSelfReferentialRaf:
    """Food-catalysed cores satisfy the letter of RAF but carry no heredity."""

    def test_a_food_catalysed_reaction_is_a_raf_but_not_self_referential(self):
        n = _net("abc", food=[0, 1], reactions=[(0, 1, 2)], catalysts=[{0}])
        assert max_raf(n).reactions == frozenset({0})     # a RAF by the definition
        assert is_food_catalysed(n, frozenset({0}))
        assert max_raf_strict(n).is_empty                 # but nothing self-referential

    def test_a_product_catalysed_reaction_survives_the_strict_reading(self):
        n = _net("abc", food=[0, 1], reactions=[(0, 1, 2)], catalysts=[{2}])
        assert not is_food_catalysed(n, frozenset({0}))
        assert max_raf_strict(n).reactions == frozenset({0})

    def test_strict_drops_only_the_food_catalysed_half(self):
        # r0 is food-catalysed, r1 needs its own product. Only r1 is self-referential.
        n = _net("abcd", food=[0, 1],
                 reactions=[(0, 1, 2), (0, 1, 3)], catalysts=[{0}, {3}])
        assert max_raf(n).reactions == frozenset({0, 1})
        assert max_raf_strict(n).reactions == frozenset({1})

    def test_a_sampled_strict_core_is_irreducible(self):
        net = binary_polymer(max_len=6, food_len=2, p=0.01,
                             rng=np.random.default_rng(3))
        base = max_raf_strict(net)
        assert not base.is_empty
        s = sample_irrraf(net, base.reactions, np.random.default_rng(1), strict=True)
        assert s and _refine(net, s, strict=True) == s
        assert not is_food_catalysed(net, s)
        for r in s:
            assert not _refine(net, s - {r}, strict=True), f"dropping {r} left a RAF"
