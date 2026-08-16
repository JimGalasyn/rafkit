"""C-BPM — Serra & Villani, *Entropy* 28(2), 184 (2026), §2.2.

The load-bearing test is `test_every_catalysed_reaction_has_a_matching_site`: it re-derives
the complementarity rule from the catalyst's own string, independently of the code that
assigned it. Everything else is a property from the paper.
"""
from __future__ import annotations

import numpy as np
import pytest

from rafkit import binary_polymer, complement, complementary_polymer, max_raf


def _catalysts(net):
    """Flatten the catalyst relation: each entry is a set of conjunctive GROUPS."""
    return {x for entry in net.catalysts for group in entry for x in group}


def _edges(net):
    return sum(len(group) for entry in net.catalysts for group in entry)


def _sites(mol, site_min, site_max):
    for w in range(site_min, min(site_max, len(mol)) + 1):
        for i in range(len(mol) - w + 1):
            yield mol[i:i + w]


class TestComplement:
    def test_flips_bits_and_is_an_involution(self):
        assert complement("100110") == "011001"          # the paper's own example
        assert complement(complement("10110")) == "10110"


class TestConstruction:
    def test_reaction_set_matches_the_k_model(self):
        """C-BPM changes WHICH catalyst acts, not what reactions exist."""
        c = complementary_polymer(max_len=6, rng=np.random.default_rng(0))
        k = binary_polymer(max_len=6, cleavage=True, rng=np.random.default_rng(0))
        assert c.molecules == k.molecules
        assert c.reactions == k.reactions
        assert c.directions == k.directions

    def test_no_catalysis_when_p_cat_zero(self):
        c = complementary_polymer(max_len=6, p_cat=0.0, rng=np.random.default_rng(0))
        assert all(not s for s in c.catalysts)
        assert c.catalysis_level == 0.0

    def test_only_polymers_at_least_as_long_as_a_site_catalyse(self):
        """Falls out of the construction; the paper states it as a consequence."""
        c = complementary_polymer(max_len=7, p_cat=1.0, site_min=4, site_max=4,
                                  rng=np.random.default_rng(1))
        for x in _catalysts(c):
            assert len(c.molecules[x]) >= 4

    def test_deterministic_for_a_seed(self):
        a = complementary_polymer(max_len=6, rng=np.random.default_rng(5))
        b = complementary_polymer(max_len=6, rng=np.random.default_rng(5))
        assert a.catalysts == b.catalysts

    def test_every_catalysed_reaction_has_a_matching_site(self):
        """Re-derive the rule from the catalyst's string, not from the assigning code.

        For a condensation the catalyst must hold a site whose first part complements the
        first reactant's SUFFIX and whose second part complements the second's PREFIX; for
        a cleavage, a site complementary to a segment spanning the cut in the substrate.
        """
        smin, smax = 3, 4
        c = complementary_polymer(max_len=7, p_cat=0.3, site_min=smin, site_max=smax,
                                  rng=np.random.default_rng(3))
        checked = 0
        for r, cats in enumerate(c.catalysts):
            for group in cats:
                for x in group:
                    cat = c.molecules[x]
                    a, b, ab = (c.molecules[i] for i in c.reactions[r])
                    if c.directions[r] > 0:                       # condensation a + b -> ab
                        ok = any(a.endswith(complement(s[:k])) and
                                 b.startswith(complement(s[k:]))
                                 for s in _sites(cat, smin, smax)
                                 for k in range(1, len(s)))
                    else:                                          # cleavage ab -> a + b
                        cut = len(a)
                        ok = any(complement(s) in
                                 {ab[cut - k: cut - k + len(s)] for k in range(1, len(s))}
                                 for s in _sites(cat, smin, smax))
                    assert ok, f"{cat!r} catalyses {a!r}+{b!r}->{ab!r} with no matching site"
                    checked += 1
        assert checked > 100, f"only {checked} assignments exercised"


class TestPaperSignature:
    def test_far_fewer_catalysts_each_doing_far_more(self):
        """Their headline contrast: ~400 reactions per C-catalyst against ~20 for K."""
        c = complementary_polymer(max_len=8, p_cat=0.05, rng=np.random.default_rng(0))
        k = binary_polymer(max_len=8, p=0.003, cleavage=True, rng=np.random.default_rng(0))
        per = lambda n: _edges(n) / max(len(_catalysts(n)), 1)
        assert per(c) > 5 * per(k)
        assert len(_catalysts(c)) < len(_catalysts(k)) / 5

    def test_hosts_a_raf(self):
        c = complementary_polymer(max_len=8, p_cat=0.05, rng=np.random.default_rng(0))
        assert len(max_raf(c).reactions) > 0


class TestValidation:
    @pytest.mark.parametrize("kw", [dict(p_cat=1.5), dict(p_cleave=-0.1)])
    def test_probabilities_checked(self, kw):
        with pytest.raises(ValueError, match="is a probability"):
            complementary_polymer(max_len=5, **kw)

    def test_site_bounds_checked(self):
        with pytest.raises(ValueError, match="site_min <= site_max"):
            complementary_polymer(max_len=5, site_min=4, site_max=3)
