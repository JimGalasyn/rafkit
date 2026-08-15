"""Stochastic simulation of a catalytic reaction network (Gillespie direct method).

The point of this module is the one thing the rest of this library cannot express:
**a subRAF has to be seeded before it can run.** A reaction catalysed by its own
product cannot start until that product appears by an *uncatalysed* event, so a maximal
RAF does not switch on -- it assembles as an order-dependent sequence of rare events.
That is Hordijk & Steel, "Autocatalytic sets extended: dynamics, inhibition, and a
generalization" (*J. Syst. Chem.* 3, 5, 2012), and `examples/` reproduces it.

Everything here is deliberately small. Rates are mass-action with unit kinetic constants,
because the published reference assigns no others and inventing them would make the
reproduction unfalsifiable. The one non-unit constant is the **uncatalysed rate
reduction factor**, which is the mechanism under test.

Conventions, all inherited from the reference rather than chosen here:

* a reaction whose catalyst is absent still proceeds, at ``1 / uncatalysed_factor``;
* a reaction with an **inhibitor present does not proceed at all** -- inhibition is a
  block, not a slowdown, and it is independent of catalysis;
* food molecules are replenished when they fall below ``food_floor``;
* a ligation ``a + b -> ab`` with ``a == b`` takes the pair count ``n(n-1)/2``, not ``n^2``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rafkit.catalysis import is_catalysed

UNCATALYSED_FACTOR = 20.0   # Hordijk & Steel (2012): "a small reduction factor of 20"
FOOD_FLOOR = 5              # "replenished when they fall below a concentration of five"


@dataclass
class Trajectory:
    """Recorded history of one run."""

    times: np.ndarray                    # (n_samples,)
    counts: np.ndarray                   # (n_samples, n_molecules)
    molecules: tuple[str, ...]
    first_fired: dict[int, float] = field(default_factory=dict)
    """Reaction index -> time it first fired, in either direction."""
    first_uncatalysed: dict[int, float] = field(default_factory=dict)
    """Reaction index -> time it first fired *without* its catalyst present.

    This is the seeding record, and the reason the module exists: a subRAF that is
    catalysed only by its own products appears here before it appears anywhere else.
    """
    first_appearance: dict[int, float] = field(default_factory=dict)
    """Molecule index -> time it first existed, at EVENT resolution.

    Deliberately not read off `counts`, which is sampled every `sample_every` events
    and therefore aliases any species that is produced and consumed between samples.
    An early version of this module inferred appearance order from the samples and
    reported a molecule appearing before its own reactant.
    """

    def of(self, name: str) -> np.ndarray:
        """Count trace for a molecule, by name."""
        return self.counts[:, self.molecules.index(name)]

    def first_seen(self, name: str) -> float | None:
        """When a molecule first existed, at event resolution; None if it never did."""
        return self.first_appearance.get(self.molecules.index(name))


def _pair_count(counts: np.ndarray, reactants: tuple[int, ...]) -> float:
    """Number of distinct reactant combinations available.

    Distinct species multiply; a species reacting with itself takes n(n-1)/2, which is
    the standard mass-action combinatorial factor and not merely n^2 -- a molecule
    cannot react with itself.
    """
    if len(reactants) == 1:
        return float(counts[reactants[0]])
    a, b = reactants
    if a == b:
        n = counts[a]
        return float(n * (n - 1) / 2)
    return float(counts[a] * counts[b])


def propensities(net, counts: np.ndarray, *,
                 uncatalysed_factor: float = UNCATALYSED_FACTOR,
                 reactions=None) -> np.ndarray:
    """Propensity of every reaction under the current counts.

    A reaction with at least one catalyst present runs at the full rate; one with none
    runs at ``1 / uncatalysed_factor`` of it. That difference is the whole mechanism:
    it makes seeding rare but not impossible.

    **Inhibition is absolute**: if any molecule inhibiting a reaction is present, its
    propensity is zero regardless of catalysis. This is what lets a running network
    *lose* a subRAF rather than only gain one -- the effect Hordijk, Naylor, Krasnogor
    & Fellermann (2018) report as toxic elements causing loss of autocatalytic subsets.

    `reactions` restricts which reactions may fire. **This is a fidelity requirement,
    not a convenience.** Hordijk & Steel study "the molecular flow on this maximal RAF";
    simulating the entire generated network instead lets any reaction fire uncatalysed,
    so species arrive by routes outside the set under study and the seeding sequence the
    experiment exists to observe is destroyed. An earlier version of this module did
    exactly that, and produced a molecule before the only reaction that makes it.
    """
    out = np.zeros(net.n_reactions)
    allowed = range(net.n_reactions) if reactions is None else reactions
    present = frozenset(np.flatnonzero(counts).tolist())
    inhibitors = getattr(net, "inhibitors", ())
    for r in allowed:
        combos = _pair_count(counts, net.reactants(r))
        if combos <= 0:
            continue
        if inhibitors and (inhibitors[r] & present):
            continue                      # inhibited: blocked outright
        catalysed = is_catalysed(net.catalysts[r], present)
        out[r] = combos if catalysed else combos / uncatalysed_factor
    return out


def simulate(net, *, n_events: int = 25_000, rng=None,
             uncatalysed_factor: float = UNCATALYSED_FACTOR,
             food_floor: int = FOOD_FLOOR, initial_food: int | None = None,
             sample_every: int = 25, reactions=None) -> Trajectory:
    """Run the direct method for `n_events` reaction events.

    Starts from food only, which is the point: everything else has to be made, and the
    parts of the network that catalyse their own production have to be seeded by an
    uncatalysed event first.

    `reactions` restricts the reaction set -- pass a maximal RAF to reproduce the
    published experiment; see `propensities` for why the default of "everything" is
    the wrong choice for that purpose.
    """
    rng = rng if rng is not None else np.random.default_rng()
    counts = np.zeros(net.n_molecules, dtype=np.int64)
    food = np.array(sorted(net.food), dtype=int)
    counts[food] = initial_food if initial_food is not None else food_floor

    times, samples = [0.0], [counts.copy()]
    first_fired: dict[int, float] = {}
    first_uncat: dict[int, float] = {}
    first_seen: dict[int, float] = {int(f): 0.0 for f in food}
    t = 0.0

    for step in range(n_events):
        a = propensities(net, counts, uncatalysed_factor=uncatalysed_factor,
                         reactions=reactions)
        a0 = a.sum()
        if a0 <= 0:
            break                                   # nothing can fire; state is dead
        t += float(rng.exponential(1.0 / a0))
        r = int(rng.choice(net.n_reactions, p=a / a0))

        if r not in first_fired:
            first_fired[r] = t
        if r not in first_uncat and not is_catalysed(
                net.catalysts[r], frozenset(np.flatnonzero(counts).tolist())):
            first_uncat[r] = t

        for x in net.reactants(r):
            counts[x] -= 1
        for x in net.products(r):
            counts[x] += 1
            first_seen.setdefault(int(x), t)
        counts[food] = np.maximum(counts[food], food_floor)   # replenish

        if (step + 1) % sample_every == 0:
            times.append(t)
            samples.append(counts.copy())

    return Trajectory(times=np.array(times), counts=np.array(samples),
                      molecules=net.molecules, first_fired=first_fired,
                      first_uncatalysed=first_uncat, first_appearance=first_seen)


def catalytically_reachable(net, reactions=None) -> frozenset[int]:
    """Molecules obtainable using **only catalysed firings** -- no seeding required.

    Everything outside this set needs at least one uncatalysed (spontaneous) reaction
    before it can exist, which is what makes the assembly of a maximal RAF an
    order-dependent sequence of rare events rather than a switch.

    It is a **least fixpoint**, and the obvious cheaper definition is wrong: taking only
    the reactions with a *food* catalyst under-counts, because a reaction whose catalyst
    is itself produced by the always-on part becomes catalysed later without ever needing
    a seed. Iterating to a fixpoint is what closes that gap -- checked against simulation
    on twelve networks, where the cheaper version fails on three of them and this does
    not.

    Static counterpart of `Trajectory.first_uncatalysed`: this predicts *which* molecules
    require a seeding event, the trajectory records *when* one happened.
    """
    from rafkit.raf import _closure

    allowed = frozenset(range(net.n_reactions) if reactions is None else reactions)
    avail = frozenset(net.food)
    while True:
        enabled = frozenset(
            r for r in allowed
            if all(x in avail for x in net.reactants(r))
            and is_catalysed(net.catalysts[r], avail)
        )
        nxt = _closure(net, enabled)
        if nxt == avail:
            return avail
        avail = nxt
