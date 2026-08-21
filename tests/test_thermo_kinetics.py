"""Rate constants reaching the simulator — the seam between `thermo` and `gillespie`.

`thermo` says what the rate constants are; whether a catalyst is present at any instant is a
property of the state and belongs to `gillespie`. These tests are about the join, and two of
them are the reason it was worth building:

* the model's existing uncatalysed factor of 20 **is already** `Kinetics(1/20, 20)`, and
  reproduces its propensities identically — so catalysis was never the missing piece;
* a chemistry built from bond energies has its **stationary point at the thermodynamic
  equilibrium**, with or without a catalyst present, which is the whole claim of the module
  arriving where it can affect a trajectory.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rafkit.binary_polymer import binary_polymer
from rafkit.catalysis import normalise
from rafkit.gillespie import UNCATALYSED_FACTOR, propensities, simulate
from rafkit.thermo import (BondEnergies, Kinetics, kinetics_from_energies,
                           reaction_free_energies, reversible_pairs,
                           unpaired_catalysis, unpaired_inhibition)

ENERGIES = BondEnergies.symmetric(-2.0, -1.0, -3.0)
DG_ASSOC = 0.5


def _net(p=0.01, seed=0, paired=True, max_len=4):
    return binary_polymer(max_len=max_len, food_len=1, p=p, cleavage=True,
                          paired_catalysis=paired, rng=np.random.default_rng(seed))


def _distinct_reactant_pair(net):
    """A reversible pair whose ligation joins two DIFFERENT molecules.

    `a + a -> aa` takes the combinatorial factor `n(n-1)/2` rather than `n**2`, which shifts
    the balance point by a factor that has nothing to do with free energy. Picking `a != b`
    keeps the equilibrium test about thermodynamics.
    """
    for i, j in reversible_pairs(net):
        a, b, _ab = net.reactions[i]
        if a != b:
            return i, j
    raise AssertionError("no pair with distinct reactants")


class TestTheModelWasAlreadyInThisForm:
    """The uncatalysed factor of 20 is an enhancement of 20, and always was."""

    def test_kinetics_reproduces_the_default_propensities_TO_THE_LAST_ULP(self):
        """Not bit-identical, and the reason is worth one line rather than a loose tolerance.

        The default computes `combos / 20`; the `Kinetics` form computes
        `combos * (1/20) * 20`, and `1/20` is not exact in binary. The two differ by at most
        one ulp — measured, 2.2e-16 on 16 of 136 reactions — which is arithmetic, not a
        difference in the model. Asserting `rel=0` would be asserting something false about
        floating point.
        """
        net = _net()
        counts = np.zeros(net.n_molecules)
        counts[:6] = [5.0, 7.0, 3.0, 2.0, 4.0, 6.0]
        default = propensities(net, counts)
        k = Kinetics(k_uncat=np.full(net.n_reactions, 1.0 / UNCATALYSED_FACTOR),
                     enhancement=UNCATALYSED_FACTOR)
        got = propensities(net, counts, kinetics=k)
        assert got == pytest.approx(default, rel=1e-15)
        assert np.max(np.abs(got - default)) < 1e-15

    def test_and_so_the_default_asserts_K_eq_1_for_every_reaction(self):
        """Unit constants both ways is `ΔG = 0`, whatever the molecules are.

        Stated at the propensity level rather than the rate-constant level: the forward and
        reverse propensities balance at `n_ab = n_a*n_b` for EVERY pair, with no reference to
        any bond energy.
        """
        net = _net(p=0.0)
        i, j = _distinct_reactant_pair(net)
        a, b, ab = net.reactions[i]
        counts = np.zeros(net.n_molecules)
        counts[a], counts[b] = 3.0, 5.0
        counts[ab] = 3.0 * 5.0                       # K = 1
        p = propensities(net, counts)
        assert p[i] == pytest.approx(p[j])


class TestTheStationaryPointIsTheEquilibrium:
    """The claim that makes any of this worth wiring up."""

    def test_forward_and_reverse_balance_at_the_mass_action_equilibrium(self):
        net = _net(p=0.0)
        i, j = _distinct_reactant_pair(net)
        a, b, ab = net.reactions[i]
        kin = kinetics_from_energies(net, ENERGIES, barrier=8.0, enhancement=1.0,
                                     dg_assoc=DG_ASSOC)
        dg = reaction_free_energies(net, ENERGIES, dg_assoc=DG_ASSOC)[i]
        keq = float(np.exp(-dg))                     # rt = 1
        counts = np.zeros(net.n_molecules)
        counts[a], counts[b] = 2.0, 3.0
        counts[ab] = keq * 2.0 * 3.0
        p = propensities(net, counts, kinetics=kin)
        assert p[i] == pytest.approx(p[j], rel=1e-12)

    def test_the_balance_point_is_NOT_where_unit_constants_put_it(self):
        """Otherwise the previous test would pass for a chemistry with no free energy in it."""
        net = _net(p=0.0)
        i, j = _distinct_reactant_pair(net)
        a, b, ab = net.reactions[i]
        dg = reaction_free_energies(net, ENERGIES, dg_assoc=DG_ASSOC)[i]
        assert dg != 0.0
        counts = np.zeros(net.n_molecules)
        counts[a], counts[b] = 2.0, 3.0
        counts[ab] = float(np.exp(-dg)) * 2.0 * 3.0
        unit = propensities(net, counts)
        assert unit[i] != pytest.approx(unit[j])

    def test_a_present_catalyst_does_not_move_the_balance_point(self):
        """The headline, at the level where it could change a trajectory.

        The catalyst multiplies both directions because they share a barrier, and under
        `paired_catalysis` it is present for both entries at once. So the same counts balance
        catalysed and uncatalysed — while both propensities rise by the enhancement.
        """
        net = _net(p=0.0)
        i, j = _distinct_reactant_pair(net)
        a, b, ab = net.reactions[i]
        catalyst = next(m for m in range(net.n_molecules) if m not in (a, b, ab))
        chi = list(net.catalysts)
        chi[i] = chi[j] = normalise([catalyst])
        net = dataclasses.replace(net, catalysts=tuple(chi))

        kin = kinetics_from_energies(net, ENERGIES, barrier=8.0, enhancement=100.0,
                                     dg_assoc=DG_ASSOC)
        dg = reaction_free_energies(net, ENERGIES, dg_assoc=DG_ASSOC)[i]
        counts = np.zeros(net.n_molecules)
        counts[a], counts[b] = 2.0, 3.0
        counts[ab] = float(np.exp(-dg)) * 2.0 * 3.0

        without = propensities(net, counts, kinetics=kin)
        counts[catalyst] = 1.0
        with_cat = propensities(net, counts, kinetics=kin)

        assert with_cat[i] == pytest.approx(with_cat[j], rel=1e-12)     # still balanced
        assert with_cat[i] == pytest.approx(without[i] * 100.0)         # and both 100x faster
        assert with_cat[j] == pytest.approx(without[j] * 100.0)


class TestUnpairedCatalysisIsThermodynamicallyImpossible:
    """`paired_catalysis=False` is not a variant chemistry; it is an impossible one."""

    def test_the_default_chemistry_is_paired(self):
        assert unpaired_catalysis(_net(p=0.05, paired=True)) == ()

    def test_drawing_the_directions_separately_is_not(self):
        net = _net(p=0.05, paired=False)
        offenders = unpaired_catalysis(net)
        assert offenders
        i, j = offenders[0]
        assert net.catalysts[i] != net.catalysts[j]

    def test_kinetics_refuses_it_rather_than_producing_plausible_rates(self):
        with pytest.raises(ValueError, match="different catalyst sets"):
            kinetics_from_energies(_net(p=0.05, paired=False), ENERGIES, barrier=8.0,
                                   enhancement=100.0)

    def test_a_catalyst_on_one_direction_only_really_does_unbalance_it(self):
        """Why the refusal is not pedantry: the free-energy source, measured.

        Built by hand so the offending catalyst is the only difference — one molecule that
        accelerates the ligation and not the cleavage. At the equilibrium counts, the
        propensities no longer balance.
        """
        net = _net(p=0.0)
        i, j = _distinct_reactant_pair(net)
        a, b, ab = net.reactions[i]
        catalyst = next(m for m in range(net.n_molecules) if m not in (a, b, ab))
        chi = list(net.catalysts)
        chi[i] = normalise([catalyst])                       # ligation half ONLY
        lopsided = dataclasses.replace(net, catalysts=tuple(chi))
        assert unpaired_catalysis(lopsided)

        kin = kinetics_from_energies(net, ENERGIES, barrier=8.0, enhancement=100.0,
                                     dg_assoc=DG_ASSOC)      # lawful rates...
        dg = reaction_free_energies(net, ENERGIES, dg_assoc=DG_ASSOC)[i]
        counts = np.zeros(net.n_molecules)
        counts[a], counts[b] = 2.0, 3.0
        counts[ab] = float(np.exp(-dg)) * 2.0 * 3.0
        counts[catalyst] = 1.0
        p = propensities(lopsided, counts, kinetics=kin)      # ...on an unlawful chemistry
        assert p[i] == pytest.approx(p[j] * 100.0)            # 100x net flux, from nothing


class TestTheSimulatorStillWorks:
    """Everything except the rate constants must be untouched."""

    def test_a_run_with_kinetics_produces_a_trajectory(self):
        net = _net(p=0.05, max_len=4)
        kin = kinetics_from_energies(net, ENERGIES, barrier=8.0, enhancement=100.0,
                                     dg_assoc=DG_ASSOC)
        traj = simulate(net, n_events=2_000, kinetics=kin,
                        rng=np.random.default_rng(7))
        assert traj.counts.shape[1] == net.n_molecules
        assert traj.times[-1] > 0.0
        assert traj.first_appearance                       # something got made

    def test_only_the_rate_constants_differ_from_a_default_run(self):
        """A `Kinetics` equal to the default reproduces a whole TRAJECTORY, not just one
        propensity vector — so the combinatorics, inhibition, food floor and seeding record
        are all provably untouched."""
        net = _net(p=0.05, max_len=4)
        kin = Kinetics(k_uncat=np.full(net.n_reactions, 1.0 / UNCATALYSED_FACTOR),
                       enhancement=UNCATALYSED_FACTOR)
        a = simulate(net, n_events=1_500, rng=np.random.default_rng(3))
        b = simulate(net, n_events=1_500, rng=np.random.default_rng(3), kinetics=kin)
        assert np.array_equal(a.counts, b.counts)   # identical event sequence despite the ulp
        assert a.first_fired == b.first_fired
        assert a.first_uncatalysed == b.first_uncatalysed
        # ⚠ Times are NOT expected to match: rescaling every rate constant by 1/20 and
        # every catalysed one back up leaves the ordering identical but the clock is set by
        # the total propensity, which the two forms reach by different arithmetic.
        assert a.times.shape == b.times.shape

    def test_passing_both_rate_specifications_is_refused(self):
        net = _net()
        kin = Kinetics(k_uncat=np.ones(net.n_reactions), enhancement=2.0)
        with pytest.raises(ValueError, match="not both"):
            propensities(net, np.ones(net.n_molecules), kinetics=kin,
                         uncatalysed_factor=50.0)

    def test_kinetics_of_the_wrong_size_is_refused(self):
        net = _net()
        with pytest.raises(ValueError, match="covers 3 reactions"):
            propensities(net, np.ones(net.n_molecules),
                         kinetics=Kinetics(k_uncat=np.ones(3), enhancement=2.0))


class TestKineticsGuards:

    def test_a_zero_uncatalysed_rate_is_refused_as_enablement(self):
        with pytest.raises(ValueError, match="enablement"):
            Kinetics(k_uncat=np.array([1.0, 0.0]), enhancement=2.0)

    def test_a_non_positive_enhancement_is_refused(self):
        with pytest.raises(ValueError, match="enhancement must be positive"):
            Kinetics(k_uncat=np.ones(2), enhancement=0.0)

    def test_k_uncat_must_be_one_dimensional(self):
        with pytest.raises(ValueError, match="one value per reaction"):
            Kinetics(k_uncat=np.ones((2, 2)), enhancement=1.0)

    def test_a_per_reaction_enhancement_must_match_k_uncat(self):
        with pytest.raises(ValueError, match="enhancement has shape"):
            Kinetics(k_uncat=np.ones(4), enhancement=np.ones(3))

    def test_rates_applies_the_enhancement_only_where_catalysed(self):
        kin = Kinetics(k_uncat=np.array([1.0, 2.0]), enhancement=10.0)
        assert kin.rates([True, False]) == pytest.approx([10.0, 2.0])
        assert kin.n_reactions == 2

    def test_rates_refuses_a_mask_of_the_wrong_length(self):
        with pytest.raises(ValueError, match="catalysed has shape"):
            Kinetics(k_uncat=np.ones(2), enhancement=1.0).rates([True])

    def test_the_enhancement_may_be_per_reaction(self):
        kin = Kinetics(k_uncat=np.ones(2), enhancement=np.array([10.0, 100.0]))
        assert kin.rates([True, True]) == pytest.approx([10.0, 100.0])

    def test_a_pair_catalysed_by_DIFFERENT_molecules_violates_too(self):
        """The row that gets missed when you count one-sided pairs.

        Forward catalysed by `{x}` and backward by `{y}` looks symmetric. It is not: a state
        holding `x` and not `y` enhances the forward direction alone. Only IDENTICAL catalyst
        sets are safe, which is why `unpaired_catalysis` tests `!=` rather than emptiness —
        and why the violating share of an unpaired chemistry is 63%, not the 47% that counting
        one-sided pairs gives.
        """
        net = _net(p=0.0)
        i, j = _distinct_reactant_pair(net)
        a, b, ab = net.reactions[i]
        spare = [m for m in range(net.n_molecules) if m not in (a, b, ab)][:2]
        chi = list(net.catalysts)
        chi[i] = normalise([spare[0]])            # forward catalyst
        chi[j] = normalise([spare[1]])            # backward catalyst -- BOTH sides catalysed
        net = dataclasses.replace(net, catalysts=tuple(chi))

        assert net.catalysts[i] and net.catalysts[j]      # neither side is bare
        assert (i, j) in unpaired_catalysis(net)          # ...and it is still caught

    def test_the_violating_share_of_an_unpaired_chemistry_is_about_two_thirds(self):
        """The number that belongs in the docstring, re-measured so it cannot drift."""
        shares = []
        for seed in range(5):
            net = binary_polymer(max_len=6, food_len=2, p=4e-3, cleavage=True,
                                 paired_catalysis=False, rng=np.random.default_rng(seed))
            shares.append(len(unpaired_catalysis(net)) / len(reversible_pairs(net)))
        assert np.mean(shares) == pytest.approx(0.63, abs=0.03)


class TestReviewFindingsOnTheWiring:
    """Each failed before the review that found it. Reproductions kept as the tests."""

    def test_the_documented_canonical_example_actually_runs(self):
        """It appeared in five places and raised ValueError verbatim.

        `k_uncat` is per reaction, so a bare scalar cannot say how many there are. `uniform`
        takes the count, which makes the example runnable instead of illustrative — and the
        five copies now quote this form.
        """
        net = _net()
        kin = Kinetics.uniform(net.n_reactions, 1 / UNCATALYSED_FACTOR, UNCATALYSED_FACTOR)
        counts = np.zeros(net.n_molecules)
        counts[:6] = [5.0, 7.0, 3.0, 2.0, 4.0, 6.0]
        assert propensities(net, counts, kinetics=kin) == pytest.approx(
            propensities(net, counts), rel=1e-15)

    def test_a_bare_scalar_still_refused_but_now_says_what_to_use(self):
        with pytest.raises(ValueError, match="Kinetics.uniform"):
            Kinetics(k_uncat=1 / 20, enhancement=20)

    def test_a_SELF_ligation_does_not_balance_where_the_docstring_said(self):
        """`a + a -> aa` takes `n_a(n_a-1)/2`, so the balance is not `n_aa/n_a^2 = K`.

        The invariant was stated unconditionally. Under unit constants (`K = 1`) at `n_a = 20`,
        balance is at `n_aa = 190`, not 400 — a factor of ~2 from where the same ΔG puts a
        hetero-ligation. The suite's own helper sidesteps `a == b`; a caller reading counts off
        a trajectory cannot.
        """
        net = _net(p=0.0)
        i, j = next((i, j) for i, j in reversible_pairs(net)
                    if net.reactions[i][0] == net.reactions[i][1])
        a, _b, ab = net.reactions[i]
        counts = np.zeros(net.n_molecules)
        counts[a] = 20.0

        counts[ab] = 20.0 * 20.0                       # what n_ab/(n_a*n_b) = K would predict
        naive = propensities(net, counts)
        assert naive[i] != pytest.approx(naive[j])     # ...and it is NOT balanced

        counts[ab] = 20.0 * 19.0 / 2.0                 # the combinatorial factor
        right = propensities(net, counts)
        assert right[i] == pytest.approx(right[j])
        assert (20.0 * 19.0 / 2.0) / 20.0 ** 2 == pytest.approx(0.475, abs=0.01)   # ~K/2

    @pytest.mark.parametrize("bad", [
        dict(k_uncat=np.array([np.inf, 1.0]), enhancement=2.0),
        dict(k_uncat=np.ones(2), enhancement=np.array([np.inf, 1.0])),
        dict(k_uncat=np.array([np.nan, 1.0]), enhancement=2.0),
    ])
    def test_a_non_finite_rate_is_refused(self, bad):
        """`inf > 0` passes a positivity test. It then makes every sampling probability NaN
        and the simulator raises somewhere else entirely — `BondEnergies` guards this for the
        same reason."""
        with pytest.raises(ValueError, match="finite"):
            Kinetics(**bad)

    def test_an_explicit_default_uncatalysed_factor_is_no_longer_silently_accepted(self):
        """The guard compared against the live default, so it could not tell `20.0` passed
        from nothing passed. `simulate` forwarded unconditionally and worked only by that
        coincidence — changing the default later would have broken every kinetics run."""
        net = _net()
        kin = Kinetics.uniform(net.n_reactions, 1.0, 2.0)
        for uf in (UNCATALYSED_FACTOR, 21.0):
            with pytest.raises(ValueError, match="not both"):
                propensities(net, np.ones(net.n_molecules), kinetics=kin,
                             uncatalysed_factor=uf)

    def test_simulate_still_forwards_correctly_under_the_new_sentinel(self):
        net = _net(p=0.05)
        kin = kinetics_from_energies(net, ENERGIES, barrier=8.0, enhancement=100.0,
                                     dg_assoc=DG_ASSOC)
        traj = simulate(net, n_events=500, kinetics=kin, rng=np.random.default_rng(1))
        assert traj.times[-1] > 0.0
        # and the plain path is unchanged by the sentinel
        a = simulate(net, n_events=500, rng=np.random.default_rng(1))
        b = simulate(net, n_events=500, rng=np.random.default_rng(1),
                     uncatalysed_factor=UNCATALYSED_FACTOR)
        assert np.array_equal(a.counts, b.counts)

    def test_asymmetric_INHIBITION_is_refused_too(self):
        """Inhibition is an absolute block, so an inhibitor on one direction only makes that
        direction impossible while the other fires — `k = 0` on half a pair, residual `inf`."""
        net = binary_polymer(max_len=4, food_len=1, p=0.01, cleavage=True, q=0.05,
                             paired_catalysis=False, rng=np.random.default_rng(1))
        assert unpaired_inhibition(net)
        # with symmetric catalysts but asymmetric inhibitors, the inhibitor check must fire
        chi = net.catalysts[:net.n_reactions // 2]
        sym = dataclasses.replace(net, catalysts=chi + chi)
        assert unpaired_catalysis(sym) == ()
        assert unpaired_inhibition(sym)
        with pytest.raises(ValueError, match="different inhibitor sets"):
            kinetics_from_energies(sym, ENERGIES, barrier=8.0, enhancement=10.0)

    def test_symmetric_inhibition_is_lawful_and_accepted(self):
        """Blocking BOTH directions is zero flux and no violation; only asymmetry breaks it."""
        net = binary_polymer(max_len=4, food_len=1, p=0.01, cleavage=True, q=0.05,
                             paired_catalysis=True, rng=np.random.default_rng(1))
        assert unpaired_inhibition(net) == ()
        kinetics_from_energies(net, ENERGIES, barrier=8.0, enhancement=10.0)

    def test_propensities_uses_Kinetics_rates_rather_than_a_second_copy(self):
        """The enhancement rule had two implementations and only the inline one was exercised.

        Checked by behaviour: monkeypatching `rates` must change what `propensities` returns.
        """
        net = _net(p=0.05)
        kin = Kinetics.uniform(net.n_reactions, 1.0, 10.0)
        counts = np.zeros(net.n_molecules)
        counts[:6] = 5.0
        base = propensities(net, counts, kinetics=kin)

        class Doubled(Kinetics):
            def rates(self, catalysed):
                return 2.0 * super().rates(catalysed)

        doubled = Doubled(k_uncat=kin.k_uncat, enhancement=kin.enhancement)
        assert propensities(net, counts, kinetics=doubled) == pytest.approx(2.0 * base)

    def test_reversible_pairs_is_computed_once_per_kinetics_build(self, monkeypatch):
        """It ran three times: here, in `unpaired_catalysis`, and in `reaction_rate_constants`."""
        import rafkit.thermo as th
        calls = []
        real = th.reversible_pairs
        monkeypatch.setattr(th, "reversible_pairs",
                            lambda net: (calls.append(1), real(net))[1])
        net = _net(p=0.01)
        th.kinetics_from_energies(net, ENERGIES, barrier=8.0, enhancement=10.0)
        assert len(calls) == 1, f"reversible_pairs called {len(calls)} times"

    def test_a_network_with_no_inhibitor_field_is_handled(self):
        """`unpaired_inhibition` is duck-typed like `gillespie.propensities`: a network type
        that never modelled inhibition has nothing to be asymmetric about."""
        net = _net(p=0.0)
        bare = dataclasses.replace(net, inhibitors=(frozenset(),) * net.n_reactions)
        object.__setattr__(bare, "inhibitors", ())      # a net that simply lacks the field
        assert unpaired_inhibition(bare) == ()
