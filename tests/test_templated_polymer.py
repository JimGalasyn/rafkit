"""Templated ligation -- catalysis by complementarity to the product, with no random draw.

The load-bearing tests are the BY-HAND ones (`TestRuleByHand`): every template of a small
chemistry written out from the definition, independently of the code that assigned it, and
`test_generator_agrees_with_the_pure_rule`, which holds the fast indexed construction to
the one-line reference. `TestKnownSizes` pins the complete ``max_len`` 7 set, whose numbers
were first obtained from a separate prototype and re-derived by an independent reviewer.
"""
from __future__ import annotations

import numpy as np
import pytest

from rafkit import (binary_polymer, catalysis_motifs, complement, degree_preserving_null,
                    is_catalysed, matched_f_cbpm, matched_f_random, max_raf,
                    motif_matched_null, templated_catalysts, templated_polymer)
from rafkit.binary_polymer import _strings


def _flat(net, r):
    """The molecules catalysing reaction `r` (each entry is a set of singleton GROUPS)."""
    return {x for group in net.catalysts[r] for x in group}


def _names(net, r):
    return {net.molecules[x] for x in _flat(net, r)}


def _ligation(net, a, b):
    index = {m: i for i, m in enumerate(net.molecules)}
    return net.reactions.index((index[a], index[b], index[a + b]))


def _degrees(net):
    n = net.n_reactions // 2
    by_molecule = np.zeros(net.n_molecules, dtype=int)
    for r in range(n):
        for x in _flat(net, r):
            by_molecule[x] += 1
    return by_molecule.tolist(), [len(_flat(net, r)) for r in range(n)]


class TestRuleByHand:
    def test_product_rule_antiparallel(self):
        # 0 + 1 -> 01; complement 10, reversed 01: every strand containing "01"
        species = tuple(_strings(3))
        got = templated_catalysts("0", "1", species, h=1)
        assert {species[i] for i in got} == {"01", "001", "010", "011", "101"}

    def test_product_rule_parallel(self):
        # the plain complement of 01 is 10
        species = tuple(_strings(3))
        got = templated_catalysts("0", "1", species, h=1, orientation="parallel")
        assert {species[i] for i in got} == {"10", "010", "100", "101", "110"}

    def test_junction_rule_reads_only_the_window(self):
        # 00 + 11 -> 0011; window a[-1] + b[0] = "01", reverse complement "01"
        species = ("01", "10", "0011", "1100", "111")
        got = templated_catalysts("00", "11", species, h=1, rule="junction")
        assert {species[i] for i in got} == {"01", "0011"}
        # the product rule asks for the whole of revcomp(0011) = 0011
        whole = templated_catalysts("00", "11", species, h=1)
        assert {species[i] for i in whole} == {"0011"}

    def test_a_reactant_shorter_than_the_overlap_is_not_templated(self):
        species = tuple(_strings(4))
        assert templated_catalysts("0", "110", species, h=2) == frozenset()
        assert templated_catalysts("011", "0", species, h=2) == frozenset()
        assert templated_catalysts("01", "10", species, h=2) != frozenset()

    def test_every_template_at_max_len_4(self):
        """h = 2 at max_len 4: the only eligible ligations are 2 + 2 -> 4, and the only
        strand that can contain a 4-residue target is the target. Sixteen reactions, one
        template each: the product's reverse complement."""
        net = templated_polymer(max_len=4, food_len=2, h=2)
        n = net.n_reactions // 2
        templated = {r for r in range(n) if net.catalysts[r]}
        assert len(templated) == 16
        for r in templated:
            a, b, ab = (net.molecules[x] for x in net.reactions[r])
            assert len(a) == len(b) == 2
            assert _names(net, r) == {complement(ab)[::-1]}
        # the four strands equal to their own reverse complement template themselves
        m = catalysis_motifs(net)
        assert (m["self_reactions"], m["self_products"]) == (4, 4)
        assert {net.molecules[net.reactions[r][2]] for r in templated
                if net.reactions[r][2] in _flat(net, r)} == {"0011", "0101", "1010", "1100"}
        assert m["pairs"] == (16 - 4) // 2


class TestConstruction:
    def test_reaction_set_matches_the_k_model(self):
        t = templated_polymer(max_len=6)
        k = binary_polymer(max_len=6, cleavage=True, rng=np.random.default_rng(0))
        assert (t.molecules, t.reactions, t.directions, t.food) == \
               (k.molecules, k.reactions, k.directions, k.food)

    def test_is_deterministic(self):
        assert templated_polymer(max_len=6) == templated_polymer(max_len=6)

    @pytest.mark.parametrize("rule", ["product", "junction"])
    @pytest.mark.parametrize("orientation", ["antiparallel", "parallel"])
    @pytest.mark.parametrize("h", [1, 2, 3])
    def test_generator_agrees_with_the_pure_rule(self, rule, orientation, h):
        net = templated_polymer(max_len=6, h=h, orientation=orientation, rule=rule)
        for r in range(net.n_reactions // 2):
            a, b, _ = (net.molecules[x] for x in net.reactions[r])
            assert _flat(net, r) == templated_catalysts(
                a, b, net.molecules, h=h, orientation=orientation, rule=rule)

    def test_the_cleavage_carries_its_ligations_templates(self):
        net = templated_polymer(max_len=6)
        n = net.n_reactions // 2
        assert net.catalysts[:n] == net.catalysts[n:]
        assert all(net.directions[r] == 1 and net.directions[r + n] == -1 for r in range(n))

    def test_catalyst_sets_are_singleton_groups(self):
        """`templated_catalysts` returns molecule indices; the network holds conjunctive
        GROUPS, each a singleton. `is_catalysed` needs the latter and raises on the former."""
        net = templated_polymer(max_len=5)
        r = _ligation(net, "01", "10")
        raw = templated_catalysts("01", "10", net.molecules)
        assert net.catalysts[r] == frozenset(frozenset({x}) for x in raw)
        assert is_catalysed(net.catalysts[r], raw)
        with pytest.raises(TypeError):
            is_catalysed(raw, raw)

    def test_orientations_differ(self):
        anti = templated_polymer(max_len=6)
        para = templated_polymer(max_len=6, orientation="parallel")
        assert anti.catalysts != para.catalysts

    def test_parallel_product_rule_has_no_self_templating(self):
        """No sequence is its own complement, so no product can contain it."""
        for h in (1, 2, 3):
            m = catalysis_motifs(templated_polymer(max_len=7, h=h, orientation="parallel"))
            assert (m["self_reactions"], m["self_products"]) == (0, 0)

    def test_template_class_restricts_who_templates(self):
        long_only = templated_polymer(max_len=6, template_class=lambda m: len(m) >= 6)
        assert all(len(long_only.molecules[x]) >= 6
                   for r in range(long_only.n_reactions) for x in _flat(long_only, r))
        nobody = templated_polymer(max_len=6, template_class=lambda m: False)
        assert not any(nobody.catalysts)
        assert max_raf(nobody).is_empty

    @pytest.mark.parametrize("kwargs", [dict(h=0), dict(orientation="sideways"),
                                        dict(rule="window"), dict(max_len=1),
                                        dict(food_len=7, max_len=7)])
    def test_refuses_what_it_cannot_mean(self, kwargs):
        with pytest.raises(ValueError):
            templated_polymer(**kwargs)


class TestKnownSizes:
    """The complete max_len 7 set: 254 species, 1,284 ligations."""

    @pytest.mark.parametrize("h, edges, self_reactions, self_products, pairs, raf", [
        (1, 9178, 54, 14, 119, 2568),
        (2, 3264, 28, 12, 114, 1568),
        (3, 574, 8, 8, 92, 0),
    ])
    def test_product_rule(self, h, edges, self_reactions, self_products, pairs, raf):
        m = catalysis_motifs(templated_polymer(max_len=7, food_len=2, h=h))
        assert (m["edges"], m["self_reactions"], m["self_products"], m["pairs"],
                m["raf_reactions"]) == (edges, self_reactions, self_products, pairs, raf)

    @pytest.mark.parametrize("h, edges, self_reactions, self_products, raf", [
        (1, 248454, 944, 240, 2568),
        (2, 35966, 256, 146, 1568),
        (3, 1590, 40, 38, 0),
    ])
    def test_junction_rule_saturates(self, h, edges, self_reactions, self_products, raf):
        m = catalysis_motifs(templated_polymer(max_len=7, food_len=2, h=h, rule="junction"))
        assert (m["edges"], m["self_reactions"], m["self_products"], m["raf_reactions"]) == \
               (edges, self_reactions, self_products, raf)
        # the same count at every product length: no specificity
        assert len({round(v, 6) for v in m["by_product_length"].values()}) == 1

    def test_product_rule_count_falls_with_product_length(self):
        """The number of superstrings up to max_len: set by max_len - L, to the few
        self-overlapping targets that have fewer (000000 has four within 7, not five)."""
        m = catalysis_motifs(templated_polymer(max_len=7, food_len=2, h=2))
        assert m["by_product_length"] == pytest.approx(
            {4: 45.875, 5: 16.625, 6: 4.96875, 7: 1.0})
        assert m["f"] == pytest.approx(12.85, abs=0.005)
        assert m["reach"] == pytest.approx(784 / 1284)
        # one more residue of room, one step along the same ladder
        m8 = catalysis_motifs(templated_polymer(max_len=8, food_len=2, h=2))
        assert m8["by_product_length"][8] == 1.0
        assert m8["by_product_length"][7] == pytest.approx(m["by_product_length"][6], abs=0.02)
        species = tuple(_strings(7))
        assert len(templated_catalysts("111", "111", species, h=2)) == 4      # target 000000
        assert len(templated_catalysts("110", "100", species, h=2)) == 5      # target 110100

    def test_mutual_pairs_are_fixed_by_the_sequence_set(self):
        """Under the product rule a mutual pair is a strand and its reverse complement."""
        for h, pairs in ((1, 119), (2, 114), (3, 92)):
            long_enough = [m for m in _strings(7) if len(m) >= 2 * h]
            own = sum(1 for m in long_enough if complement(m)[::-1] == m)
            assert (len(long_enough) - own) // 2 == pairs

    def test_h3_needs_reactants_the_uncatalysed_background_makes(self):
        net = templated_polymer(max_len=7, food_len=2, h=3)
        assert any(net.catalysts) and max_raf(net).is_empty
        # with food reaching the overlap, the same rule closes
        assert not max_raf(templated_polymer(max_len=7, food_len=3, h=3)).is_empty


@pytest.fixture(scope="module")
def net():
    return templated_polymer(max_len=7, food_len=2, h=2)


class TestNulls:
    @pytest.mark.parametrize("stratified", [False, True])
    def test_degree_preserving_keeps_every_degree_and_moves_the_edges(self, net, stratified):
        null = degree_preserving_null(net, np.random.default_rng(0), stratified=stratified)
        assert _degrees(null) == _degrees(net)
        assert (null.reactions, null.directions, null.food) == \
               (net.reactions, net.directions, net.food)
        n = net.n_reactions // 2
        assert null.catalysts[:n] == null.catalysts[n:]
        moved = sum(len(_flat(net, r) - _flat(null, r)) for r in range(n))
        assert moved > 0.8 * catalysis_motifs(net)["edges"]
        a, b = catalysis_motifs(net), catalysis_motifs(null)
        assert (b["edges"], b["reach"], b["by_product_length"]) == \
               (a["edges"], a["reach"], a["by_product_length"])
        assert b["pairs"] < a["pairs"] / 2            # which strand templates which is gone

    def test_stratified_keeps_each_reactions_template_lengths(self, net):
        def lengths(x):
            return [sorted(len(x.molecules[t]) for t in _flat(x, r))
                    for r in range(x.n_reactions // 2)]
        rng = np.random.default_rng(1)
        assert lengths(degree_preserving_null(net, rng, stratified=True)) == lengths(net)
        assert lengths(degree_preserving_null(net, rng)) != lengths(net)

    def test_plain_shuffle_frees_templates_from_the_products_length(self, net):
        """The product rule's template is never shorter than its product; the plain
        shuffle's often is -- which is what the stratified null exists to separate."""
        def shorter(x):
            return sum(len(x.molecules[t]) < len(x.molecules[x.reactions[r][2]])
                       for r in range(x.n_reactions // 2) for t in _flat(x, r))
        rng = np.random.default_rng(2)
        assert shorter(net) == 0
        assert shorter(degree_preserving_null(net, rng, stratified=True)) == 0
        assert shorter(degree_preserving_null(net, rng)) > 100

    def test_degree_preserving_is_seeded(self, net):
        a = degree_preserving_null(net, np.random.default_rng(5))
        b = degree_preserving_null(net, np.random.default_rng(5))
        c = degree_preserving_null(net, np.random.default_rng(6))
        assert a == b and a != c

    def test_rewiring_refuses_unpaired_catalysis(self):
        unpaired = binary_polymer(max_len=5, p=0.05, rng=np.random.default_rng(0),
                                  cleavage=True, paired_catalysis=False)
        with pytest.raises(ValueError, match="paired"):
            degree_preserving_null(unpaired, np.random.default_rng(0))

    def test_motif_matched_has_exactly_the_motifs_and_the_edges(self, net):
        a = catalysis_motifs(net)
        for seed in range(3):
            b = catalysis_motifs(motif_matched_null(net, np.random.default_rng(seed)))
            assert (b["edges"], b["self_reactions"], b["pairs"]) == \
                   (a["edges"], a["self_reactions"], a["pairs"])
            assert b["reach"] > a["reach"]            # nothing else of the structure is kept

    def test_motif_matched_by_length_has_the_motifs_where_the_rule_has_them(self, net):
        """Two counts are a thin description: planted anywhere, the self-catalysing products
        and the pairs land at every length. `match_lengths` puts them at the rule's."""
        a = catalysis_motifs(net)
        assert a["self_products_by_length"] == {4: 4, 6: 8}
        assert a["self_reactions_by_length"] == {4: 4, 6: 24}
        assert a["pairs_by_lengths"] == {(4, 4): 6, (5, 5): 16, (6, 6): 28, (7, 7): 64}
        keys = ("edges", "self_reactions", "pairs", "self_products", "self_products_by_length",
                "self_reactions_by_length", "pairs_by_lengths")
        for seed in range(3):
            b = catalysis_motifs(motif_matched_null(net, np.random.default_rng(seed), match_lengths=True))
            assert {k: b[k] for k in keys} == {k: a[k] for k in keys}
        loose = catalysis_motifs(motif_matched_null(net, np.random.default_rng(0)))
        assert loose["pairs_by_lengths"] != a["pairs_by_lengths"]
        assert loose["self_products"] > a["self_products"]

    def test_matched_f_random_matches_f_in_expectation(self, net):
        target = catalysis_motifs(net)["f"]
        fs = [catalysis_motifs(matched_f_random(net, np.random.default_rng(s)))["f"]
              for s in range(8)]
        assert np.mean(fs) == pytest.approx(target, rel=0.02)
        assert len(set(fs)) > 1                       # binomial, not exact

    def test_matched_f_cbpm_matches_f_in_expectation(self, net):
        target = catalysis_motifs(net)["f"]
        fs = [catalysis_motifs(matched_f_cbpm(net, np.random.default_rng(s)))["f"]
              for s in range(12)]
        assert np.mean(fs) == pytest.approx(target, rel=0.15)
        with pytest.raises(ValueError, match="p_cat"):
            matched_f_cbpm(net, np.random.default_rng(0), p_cat=0.1)

    def test_matched_ensembles_refuse_an_incomplete_set(self, net):
        ligations_only = binary_polymer(max_len=5, p=0.05, rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="complete"):
            matched_f_random(ligations_only, np.random.default_rng(0))


class TestMotifs:
    def test_counts_on_the_reversible_pair(self):
        """`f` here is `catalysis_level`: a ligation and its cleavage are one reaction."""
        net = templated_polymer(max_len=6)
        assert catalysis_motifs(net)["f"] == pytest.approx(net.catalysis_level)
        k = binary_polymer(max_len=6, p=0.01, rng=np.random.default_rng(3), cleavage=True)
        assert catalysis_motifs(k)["f"] == pytest.approx(k.catalysis_level)

    def test_an_uncatalysed_network_has_no_motifs(self):
        m = catalysis_motifs(templated_polymer(max_len=5, template_class=lambda s: False))
        assert (m["edges"], m["f"], m["reach"], m["per_reaction"], m["self_reactions"],
                m["pairs"], m["raf_reactions"]) == (0, 0.0, 0.0, 0.0, 0, 0, 0)
        assert m["by_product_length"] == {}
