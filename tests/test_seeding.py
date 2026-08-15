"""Gate 1: a maximal RAF assembles by seeding, it does not switch on.

Reproduces the mechanism reported by Hordijk & Steel, "Autocatalytic sets extended:
dynamics, inhibition, and a generalization" (*J. Syst. Chem.* 3, 5, 2012): parts of a
maximal RAF catalysed only by their own products cannot start until those products
appear by a rare **uncatalysed** reaction, so the set comes into existence as an
order-dependent sequence of events.

Their exact network is a random draw we cannot reconstruct, so these run over an
ensemble of networks generated at **their published parameters** (n=5, t=2, p=0.0045,
cleavage-ligation) and assert only what is *causally necessary*. Two phenomenological
features of their figure are deliberately not asserted here:

* a species declining once a later subRAF consumes it -- topology-specific, and it does
  not reproduce cleanly on these networks;
* growth levelling off as cleavage catches up -- it does reproduce, but needs a longer
  run than their 25,000 events, so it lives in `examples/` where run length is free.

Both were measured before being dropped, rather than quietly omitted.
"""
from __future__ import annotations

import numpy as np
import pytest

from rafkit import binary_polymer, max_raf, simulate
from rafkit.catalysis import is_catalysed
from rafkit.gillespie import catalytically_reachable

# Hordijk & Steel (2012) §"A realistic example".
PUBLISHED = dict(max_len=5, food_len=2, p=0.0045)
SEEDS = [540, 442, 1591, 1660, 44, 95]


def _run(seed, n_events=20_000):
    net = binary_polymer(**PUBLISHED, rng=np.random.default_rng(seed), cleavage=True)
    raf = sorted(max_raf(net).reactions)
    if not raf:
        pytest.skip(f"seed {seed} generated no RAF")
    return net, raf, simulate(net, n_events=n_events, rng=np.random.default_rng(0),
                              reactions=raf)


@pytest.mark.parametrize("seed", SEEDS)
def test_food_catalysed_reactions_never_need_seeding(seed):
    """Criterion 1. Food never depletes, so a reaction with a food catalyst always has
    one: it can run from t=0 and can never fire uncatalysed. Exact, not statistical."""
    net, raf, tr = _run(seed)
    food_catalysed = [r for r in raf if is_catalysed(net.catalysts[r], net.food)]
    assert food_catalysed, "no always-on reactions in this network"
    for r in food_catalysed:
        assert r not in tr.first_uncatalysed, (
            f"reaction {r} has a food catalyst yet fired uncatalysed")


@pytest.mark.parametrize("seed", SEEDS)
def test_nothing_beyond_the_catalytic_core_appears_before_a_seeding_event(seed):
    """Criterion 2, and the mechanism itself: molecules that catalysed firings alone
    cannot reach must wait for a spontaneous reaction."""
    net, raf, tr = _run(seed)
    reachable = catalytically_reachable(net, raf)
    beyond = {m: t for m, t in tr.first_appearance.items() if m not in reachable}
    if not beyond:
        pytest.skip("this network needs no seeding; nothing to test")

    assert tr.first_uncatalysed, "molecules appeared beyond the core with no seeding event"
    first_seed = min(tr.first_uncatalysed.values())
    earliest, t = min(beyond.items(), key=lambda kv: kv[1])
    assert t >= first_seed, (
        f"{net.molecules[earliest]} appeared at {t:.4f}, before the first seeding "
        f"event at {first_seed:.4f}")


def test_the_static_prediction_is_tighter_than_the_naive_one():
    """`catalytically_reachable` iterates to a fixpoint. Taking only food-catalysed
    reactions is the tempting cheaper version and it under-counts, because a reaction
    catalysed by something the always-on part makes becomes catalysed without a seed."""
    from rafkit.raf import _closure
    net = binary_polymer(**PUBLISHED, rng=np.random.default_rng(95), cleavage=True)
    raf = sorted(max_raf(net).reactions)
    naive = _closure(net, frozenset(r for r in raf
                                    if is_catalysed(net.catalysts[r], net.food)))
    assert naive < catalytically_reachable(net, raf)
