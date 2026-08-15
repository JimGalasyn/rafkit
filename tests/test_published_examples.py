"""Worked examples from the literature, with their published answers.

Every case here is a system somebody else wrote down and stated the answer for, so a
failure means this library disagrees with the field rather than with our expectations.
That is the strongest kind of test available, and cheaper than it looks -- these are all
five reactions or fewer.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from rafkit import max_raf, parse_crs, sample_irrraf
from rafkit.raf import _refine, core_raf, has_unique_irraf


def _named(net, reactions):
    return frozenset(net.names[r] for r in reactions)


def _all_raf_subsets(net):
    n = net.n_reactions
    return [s for k in range(1, n + 1) for c in combinations(range(n), k)
            if (s := frozenset(c)) and _refine(net, s) == s]


# ---------------------------------------------------------------------------
# Huson, Xavier & Steel, J. R. Soc. Interface 21(214):20230732 (2024),
# the illustration of corollary 3.1: "based on Kauffman's binary polymer model
# with food set F = {0,1,00,01,10,11}".
KAUFFMAN_BPM = """
Food: 0, 1, 00, 01, 10, 11
r1 : 10 + 0    [01100] => 100
r2 : 01 + 100  [0]     => 01100
r3 : 10 + 1    [0]     => 101
r4 : 11 + 10   [101]   => 1110
r5 : 1110 + 0  [101]   => 11100
"""


class TestKauffmanBinaryPolymerExample:
    """"This system is itself an RAF and it contains six other RAFs as subsets."""

    def test_the_whole_system_is_a_raf(self):
        net = parse_crs(KAUFFMAN_BPM)
        assert _named(net, max_raf(net).reactions) == {"r1", "r2", "r3", "r4", "r5"}

    def test_exactly_the_seven_published_rafs(self):
        net = parse_crs(KAUFFMAN_BPM)
        got = {frozenset(net.names[r] for r in s) for s in _all_raf_subsets(net)}
        assert got == {
            frozenset({"r3"}),
            frozenset({"r1", "r2"}),
            frozenset({"r3", "r4"}),
            frozenset({"r1", "r2", "r3"}),
            frozenset({"r3", "r4", "r5"}),
            frozenset({"r1", "r2", "r3", "r4"}),
            frozenset({"r1", "r2", "r3", "r4", "r5"}),
        }

    def test_the_irreducible_ones_are_r1r2_and_r3(self):
        net = parse_crs(KAUFFMAN_BPM)
        cores = {_named(net, sample_irrraf(net, max_raf(net).reactions,
                                           np.random.default_rng(i)))
                 for i in range(50)}
        assert cores == {frozenset({"r1", "r2"}), frozenset({"r3"})}

    def test_two_irrafs_means_the_core_test_says_not_unique(self):
        net = parse_crs(KAUFFMAN_BPM)
        assert not has_unique_irraf(net)


# ---------------------------------------------------------------------------
# Same paper, §4.1: three iRAFs, none nested, "each one is a subset of the union
# of the two others" -- the case that makes extending pairwise results to three fail.
THREE_IRRAFS = """
Food: f
r1 : f [x1] => x2 + x3
r2 : f [x2] => x1 + x3
r3 : f [x3] => x1 + x2
"""


class TestThreeIrreducibleRafs:
    def test_finds_all_three_published_irrafs(self):
        net = parse_crs(THREE_IRRAFS)
        cores = {_named(net, sample_irrraf(net, max_raf(net).reactions,
                                           np.random.default_rng(i)))
                 for i in range(60)}
        assert cores == {frozenset({"r1", "r2"}), frozenset({"r1", "r3"}),
                         frozenset({"r2", "r3"})}

    def test_no_irraf_is_nested_in_another(self):
        net = parse_crs(THREE_IRRAFS)
        cores = list({sample_irrraf(net, max_raf(net).reactions,
                                    np.random.default_rng(i)) for i in range(60)})
        for a in cores:
            for b in cores:
                assert a == b or not (a < b)

    def test_each_is_contained_in_the_union_of_the_other_two(self):
        """The published subtlety: pairwise non-nesting does not extend to triples."""
        net = parse_crs(THREE_IRRAFS)
        cores = list({sample_irrraf(net, max_raf(net).reactions,
                                    np.random.default_rng(i)) for i in range(60)})
        assert len(cores) == 3
        for i, c in enumerate(cores):
            others = cores[:i] + cores[i + 1:]
            assert c <= others[0] | others[1]


class TestCoreRaf:
    def test_core_is_the_iraf_when_it_is_unique(self):
        # Mutually dependent pair: neither reaction survives without the other, so the
        # maximal RAF is already irreducible and Core must equal it.
        net = parse_crs("Food: a, b\nr1 : a + b [d] => c\nr2 : a + c [c] => d\n")
        assert has_unique_irraf(net)
        assert core_raf(net) == max_raf(net).reactions

    def test_no_raf_is_not_a_unique_iraf(self):
        net = parse_crs("Food: a, b\nr1 : a + b [z] => c\n")
        assert max_raf(net).is_empty
        assert not has_unique_irraf(net)
