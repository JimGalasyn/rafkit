"""The firing-disk construction — Serra & Villani, *Entropy* 28(2), 184 (2026), §2.2.

The load-bearing property is `test_every_species_is_produced_or_seed`: a grown chemistry
is closed under its own production, which is exactly what distinguishes it from an
enumerated one and is the reason it exists here.
"""
from __future__ import annotations

import numpy as np
import pytest

from rafkit import firing_disk_polymer, max_raf


class TestGrowth:
    def test_every_species_is_produced_or_seed(self):
        """No unreachable species: each molecule is food, or some reaction makes it."""
        net = firing_disk_polymer(rng=np.random.default_rng(0))
        made = {m for r in range(net.n_reactions) for m in net.products(r)}
        for x in range(net.n_molecules):
            assert x in net.food or x in made, f"{net.molecules[x]!r} unreachable"

    def test_disk_is_the_food_set(self):
        net = firing_disk_polymer(disk_size=16, disk_len=4,
                                  rng=np.random.default_rng(1))
        assert len(net.food) == 16
        assert all(len(net.molecules[x]) <= 4 for x in net.food)

    def test_respects_max_len(self):
        net = firing_disk_polymer(max_len=6, rng=np.random.default_rng(2))
        assert all(len(m) <= 6 for m in net.molecules)

    def test_no_seed_catalysts_means_no_growth(self):
        """Nothing can fire, so the chemistry is the disk and there are no reactions."""
        net = firing_disk_polymer(n_cleave_cat=0, n_cond_cat=0, p_cat=0.0,
                                  disk_size=20, rng=np.random.default_rng(3))
        assert net.n_reactions == 0
        assert net.n_molecules == 20

    def test_deterministic_for_a_seed(self):
        a = firing_disk_polymer(max_len=8, rng=np.random.default_rng(7))
        b = firing_disk_polymer(max_len=8, rng=np.random.default_rng(7))
        assert a.molecules == b.molecules and a.catalysts == b.catalysts

    def test_reactions_are_consistent_splits(self):
        net = firing_disk_polymer(max_len=8, rng=np.random.default_rng(4))
        for a, b, ab in net.reactions:
            assert net.molecules[a] + net.molecules[b] == net.molecules[ab]


class TestPaperEnsemble:
    """Figure 3's ensemble: ~2000 species, ~40,000 reactions, ~100 catalysts."""

    def test_reaches_the_published_scale(self):
        net = firing_disk_polymer(rng=np.random.default_rng(0))       # paper defaults
        assert 1500 <= net.n_molecules <= 2100
        assert 20_000 <= net.n_reactions <= 45_000
        n_cat = len({x for e in net.catalysts for g in e for x in g})
        assert 40 <= n_cat <= 200, n_cat

    def test_raf_is_nearly_the_whole_chemistry(self):
        """Their large C-chemistries host 'a RAF often as large as the entire chemistry'."""
        net = firing_disk_polymer(rng=np.random.default_rng(0))
        assert len(max_raf(net).reactions) > 0.8 * net.n_reactions


class TestValidation:
    def test_probabilities_checked(self):
        with pytest.raises(ValueError, match="is a probability"):
            firing_disk_polymer(p_cat=2.0)

    def test_site_bounds_checked(self):
        with pytest.raises(ValueError, match="site_min <= site_max"):
            firing_disk_polymer(site_min=5, site_max=3)
