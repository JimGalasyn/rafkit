"""Free energy for a polymer chemistry — the two exact anchors, and the traps around them.

These are deterministic, closed-form checks. The module's claims are algebraic (a state
function, detailed balance, a 1-D Ising transfer matrix, Flory's distribution), so every one
of them can be hit exactly rather than sampled, and the tests are written to hit them exactly.

Two checks are layered on purpose. The equilibrium weight of a sequence is computed here by
**brute force from mass action** — `exp(−(G(s) + (L−1)·dg_assoc)/RT)` times the monomer
concentrations — with no reference to `transfer_matrix`, and the transfer matrix is then
required to reproduce it. Only after that is the matrix used to reach lengths brute force
cannot. A test that used the matrix to check the matrix would pass on a wrong normalisation.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from rafkit.binary_polymer import binary_polymer
from rafkit.thermo import (RT_KJ_PER_MOL_298K, BondEnergies, Rates,
                           barrier_drop_from_enhancement, catalysed_rates,
                           detailed_balance_residual, elongation_ratio,
                           enhancement_from_barrier_drop, equilibrium_constant,
                           mean_length, rate_constants, reaction_free_energies,
                           reaction_rate_constants, reversible_pairs,
                           sequence_correlation_length, transfer_matrix)

# A deliberately NON-additive assignment: eps = -2 + -3 - 2*(-1) = -3, so it has a genuine
# sequence preference and is the arm most of these tests run in.
NONADDITIVE = BondEnergies.symmetric(-2.0, -1.0, -3.0)
# eps = -2 + -3 - 2*(-2.5) = 0 exactly: additive but NOT uniform, which is the case that
# separates "no sequence preference" from "one bond energy".
ADDITIVE = BondEnergies.symmetric(-2.0, -2.5, -3.0)
UNIFORM = BondEnergies.uniform(-1.0)


def _sequences(max_len: int, alphabet: str = "01"):
    for n in range(1, max_len + 1):
        for bits in product(alphabet, repeat=n):
            yield "".join(bits)


def _mass_action_weight(s: str, energies: BondEnergies, *, monomer, dg_assoc: float,
                        rt: float) -> float:
    """Equilibrium concentration of one sequence, from mass action and nothing else.

    Building `s` from its monomers takes `len(s) - 1` ligations whose free energies sum to
    `G(s) + (L-1)*dg_assoc`, so `[s] = exp(-that/RT) * prod([monomer_i])`. Deliberately
    written without `transfer_matrix`, which is the thing it exists to check.
    """
    m = np.broadcast_to(np.asarray(monomer, dtype=float), (len(energies.alphabet),))
    conc = float(np.prod([m[energies.alphabet.index(c)] for c in s]))
    return conc * float(np.exp(-(energies.sequence(s) + (len(s) - 1) * dg_assoc) / rt))


class TestTheStateFunction:
    """ΔG of a ligation is the junction bond alone, and that must be a state function."""

    def test_a_monomer_has_no_bonds(self):
        assert NONADDITIVE.sequence("0") == 0.0
        assert NONADDITIVE.sequence("1") == 0.0

    def test_sequence_is_the_sum_over_its_bonds(self):
        # 0-1, 1-0, 0-1 = -1 + -1 + -1 under the symmetric assignment
        assert NONADDITIVE.sequence("0101") == pytest.approx(-3.0)
        # 0-0, 0-0 = -2 + -2
        assert NONADDITIVE.sequence("000") == pytest.approx(-4.0)

    @pytest.mark.parametrize("s", ["0110100", "0000000", "1010101", "01", "0011"])
    def test_every_split_agrees_with_the_whole(self, s):
        """Wegscheider consistency, stated concretely.

        `G(a) + G(b) + ligation(a, b)` must equal `G(ab)` for EVERY split — the local rule
        against the global sum. An off-by-one in the junction index passes the round trip on
        a palindrome and fails here.
        """
        whole = NONADDITIVE.sequence(s)
        for i in range(1, len(s)):
            a, b = s[:i], s[i:]
            local = NONADDITIVE.sequence(a) + NONADDITIVE.sequence(b) + NONADDITIVE.ligation(a, b)
            assert local == pytest.approx(whole, abs=1e-12), f"split at {i}"

    def test_different_splits_have_different_free_energies(self):
        """The point of a non-additive assignment: where you cut matters.

        `00` + `11` forms a 0-1 bond; `001` + `1` forms a 1-1 bond. Same product, different
        reaction free energy — which is where a preference for particular cleavage sites
        comes from, and it vanishes identically under `UNIFORM`.
        """
        assert NONADDITIVE.ligation("00", "11") == pytest.approx(-1.0)
        assert NONADDITIVE.ligation("001", "1") == pytest.approx(-3.0)
        assert UNIFORM.ligation("00", "11") == UNIFORM.ligation("001", "1")

    def test_a_cycle_of_reactions_has_zero_free_energy(self):
        """`0011` made two ways, taken apart a third: the loop must close at exactly 0.

        This is what "state function" buys, and it is the property a hand-assigned per-
        reaction ΔG would silently violate.
        """
        route_a = (NONADDITIVE.ligation("0", "0") + NONADDITIVE.ligation("00", "1")
                   + NONADDITIVE.ligation("001", "1"))
        route_b = (NONADDITIVE.ligation("1", "1") + NONADDITIVE.ligation("0", "11")
                   + NONADDITIVE.ligation("0", "011"))
        assert route_a == pytest.approx(route_b, abs=1e-12)
        assert route_a == pytest.approx(NONADDITIVE.sequence("0011"), abs=1e-12)


class TestDetailedBalanceIsStructural:
    """`k_f/k_r = exp(-ΔG/RT)` must hold for every parameter setting, not for a chosen one."""

    @pytest.mark.parametrize("beta", [0.0, 0.3, 0.5, 1.0])
    @pytest.mark.parametrize("barrier", [5.0, 20.0])
    @pytest.mark.parametrize("rt", [1.0, RT_KJ_PER_MOL_298K])
    def test_the_ratio_is_independent_of_barrier_and_beta(self, beta, barrier, rt):
        dg = np.array([-3.0, -0.5, 0.0, 0.5, 2.0]) * rt
        r = rate_constants(dg, barrier=barrier * rt, beta=beta, rt=rt)
        assert r.equilibrium_constant == pytest.approx(equilibrium_constant(dg, rt=rt))

    def test_a_catalyst_does_not_move_the_equilibrium(self):
        """The definition of a catalyst, and the thing `k_uncat = 0` violates."""
        r = rate_constants(np.array([-2.0, 1.5]), barrier=10.0)
        for factor in (1e-3, 1.0, 20.0, 100.0, 1e6):
            c = catalysed_rates(r, factor)
            assert c.equilibrium_constant == pytest.approx(r.equilibrium_constant)
            assert c.forward == pytest.approx(r.forward * factor)

    def test_scaling_only_the_forward_rate_moves_it(self):
        """Stated as a test so the failure mode is on the record, not only in a docstring."""
        r = rate_constants(np.array([-2.0]), barrier=10.0)
        wrong = Rates(forward=r.forward * 100.0, reverse=r.reverse)
        assert wrong.equilibrium_constant == pytest.approx(r.equilibrium_constant * 100.0)

    def test_rates_built_from_the_energies_have_zero_residual(self):
        net = binary_polymer(max_len=4, food_len=1, p=0.01, cleavage=True,
                             rng=np.random.default_rng(0))
        k = reaction_rate_constants(net, NONADDITIVE, barrier=6.0, dg_assoc=0.5)
        assert detailed_balance_residual(net, k, NONADDITIVE, dg_assoc=0.5) < 1e-12

    def test_unit_rate_constants_assert_that_every_reaction_is_thermoneutral(self):
        """What the library currently does, measured.

        `gillespie` gives every reaction a unit rate constant, which is `K_eq = 1` and so
        `ΔG = 0` for all of them. Against any real bond energy the residual is therefore the
        largest `|ΔG|/RT` in the network — here `|0.5 - 3.0| = 2.5`.
        """
        net = binary_polymer(max_len=4, food_len=1, p=0.01, cleavage=True,
                             rng=np.random.default_rng(0))
        residual = detailed_balance_residual(net, np.ones(net.n_reactions), NONADDITIVE,
                                             dg_assoc=0.5)
        assert residual == pytest.approx(2.5)

    def test_enablement_is_an_infinite_violation(self):
        net = binary_polymer(max_len=4, food_len=1, p=0.01, cleavage=True,
                             rng=np.random.default_rng(0))
        k = reaction_rate_constants(net, NONADDITIVE, barrier=6.0)
        k[0] = 0.0                       # the reverse of some reaction cannot happen at all
        assert detailed_balance_residual(net, k, NONADDITIVE) == float("inf")

    def test_an_irreversible_network_RAISES_rather_than_returning_zero(self):
        """Silence is not a negative.

        A ligation-only chemistry has nothing to check. Returning 0.0 — the value that means
        "consistent" — would report a clean bill of health for a network that cannot have
        one, so the absence of pairs has to be a different outcome from their agreement.
        """
        net = binary_polymer(max_len=4, food_len=1, p=0.01, cleavage=False,
                             rng=np.random.default_rng(0))
        assert reversible_pairs(net) == ()
        with pytest.raises(ValueError, match="no reversible pairs"):
            detailed_balance_residual(net, np.ones(net.n_reactions), NONADDITIVE)

    def test_pairs_are_matched_on_the_reaction_and_not_on_position(self):
        net = binary_polymer(max_len=4, food_len=1, p=0.0, cleavage=True,
                             rng=np.random.default_rng(0))
        pairs = reversible_pairs(net)
        assert len(pairs) == net.n_reactions // 2
        for i, j in pairs:
            assert net.reactions[i] == net.reactions[j]
            assert net.directions[i] > 0 > net.directions[j]

    def test_a_per_reaction_enhancement_must_match_across_a_pair(self):
        net = binary_polymer(max_len=4, food_len=1, p=0.01, cleavage=True,
                             rng=np.random.default_rng(0))
        e = np.ones(net.n_reactions)
        e[reversible_pairs(net)[0][0]] = 100.0        # the ligation half only
        with pytest.raises(ValueError, match="differs across"):
            reaction_rate_constants(net, NONADDITIVE, barrier=6.0, enhancement=e)

    def test_a_paired_enhancement_is_accepted_and_leaves_the_residual_at_zero(self):
        net = binary_polymer(max_len=4, food_len=1, p=0.01, cleavage=True,
                             rng=np.random.default_rng(0))
        e = np.ones(net.n_reactions)
        for i, j in reversible_pairs(net)[:5]:
            e[i] = e[j] = 100.0
        k = reaction_rate_constants(net, NONADDITIVE, barrier=6.0, enhancement=e)
        assert detailed_balance_residual(net, k, NONADDITIVE) < 1e-12


class TestReactionFreeEnergiesOnANetwork:
    """The stored `(a, b, ab)` triple carries the direction separately; both must be right."""

    def test_direction_flips_the_sign_and_nothing_else(self):
        net = binary_polymer(max_len=4, food_len=1, p=0.0, cleavage=True,
                             rng=np.random.default_rng(1))
        dg = reaction_free_energies(net, NONADDITIVE, dg_assoc=0.5)
        for i, j in reversible_pairs(net):
            assert dg[i] == pytest.approx(-dg[j])

    def test_every_reaction_matches_products_minus_reactants(self):
        """Computed from the junction rule inside the module, checked against the global sum.

        `dg_assoc` is added for a ligation and subtracted for a cleavage, so it is carried
        through the direction flip rather than applied afterwards.
        """
        net = binary_polymer(max_len=5, food_len=1, p=0.0, cleavage=True,
                             rng=np.random.default_rng(2))
        dg = reaction_free_energies(net, NONADDITIVE, dg_assoc=0.5)
        for r, (a, b, ab) in enumerate(net.reactions):
            g_a, g_b, g_ab = (NONADDITIVE.sequence(net.molecules[x]) for x in (a, b, ab))
            expected = (g_ab - g_a - g_b) + 0.5
            assert dg[r] == pytest.approx(expected if net.directions[r] > 0 else -expected)


class TestTheBarrier:
    """`k_uncat` becomes a consequence of the barrier, and the barrier has a floor."""

    def test_a_barrier_below_its_own_wells_is_refused(self):
        with pytest.raises(ValueError, match="below the transition state"):
            rate_constants(-10.0, barrier=1.0)          # needs barrier >= 5 at beta = 0.5
        with pytest.raises(ValueError, match="below the transition state"):
            rate_constants(10.0, barrier=1.0)

    def test_the_floor_moves_with_beta(self):
        """At beta = 0 the transition state is reactant-like, so a downhill reaction needs
        no barrier at all — but the uphill direction then needs the full ΔG."""
        rate_constants(-10.0, barrier=0.0, beta=0.0)                 # allowed
        with pytest.raises(ValueError):
            rate_constants(10.0, barrier=9.0, beta=0.0)              # needs 10

    def test_no_rate_exceeds_the_prefactor(self):
        dg = np.array([-8.0, -1.0, 0.0, 1.0, 8.0])
        r = rate_constants(dg, barrier=10.0, prefactor=3.0)
        assert np.all(r.forward <= 3.0 + 1e-12)
        assert np.all(r.reverse <= 3.0 + 1e-12)

    def test_the_uncatalysed_rate_is_determined_and_never_zero(self):
        """A consistent ΔG cannot coexist with a zero uncatalysed rate — the whole point."""
        r = rate_constants(np.array([-5.0, 0.0, 5.0]), barrier=30.0)
        assert np.all(r.forward > 0.0)
        assert np.all(r.reverse > 0.0)

    def test_enhancement_and_barrier_drop_are_inverses(self):
        for x in (2.0, 20.0, 100.0, 1e6):
            assert enhancement_from_barrier_drop(
                barrier_drop_from_enhancement(x)) == pytest.approx(x)

    def test_a_hundredfold_catalyst_is_about_one_hydrogen_bond(self):
        """The scale check that makes 100x a physical claim rather than a round number."""
        drop = barrier_drop_from_enhancement(100.0, rt=RT_KJ_PER_MOL_298K)
        assert drop == pytest.approx(11.4, abs=0.05)      # kJ/mol at 298 K


class TestSequencePreference:
    """Where a thermodynamic basis for templating would have to come from."""

    def test_a_uniform_assignment_has_no_sequence_preference(self):
        assert UNIFORM.nonadditivity == 0.0
        assert sequence_correlation_length(UNIFORM) == 0.0

    def test_and_neither_does_an_ADDITIVE_one_that_is_not_uniform(self):
        """The sharper statement, and the reason `nonadditivity` exists.

        `E01 = (E00 + E11)/2` makes the bond energy `h(a) + h(b)`: it says something about
        each residue and nothing about the pair. Three energies chosen this way buy exactly
        nothing over one, which "a uniform bond energy admits no preference" does not say.
        """
        assert ADDITIVE.e[0, 0] != ADDITIVE.e[1, 1]           # genuinely not uniform
        assert ADDITIVE.nonadditivity == 0.0
        assert sequence_correlation_length(ADDITIVE) == 0.0

    def test_non_additivity_gives_a_finite_correlation_length(self):
        assert NONADDITIVE.nonadditivity == pytest.approx(-3.0)
        assert sequence_correlation_length(NONADDITIVE) > 0.5

    def test_the_sign_of_eps_says_blocks_or_alternation(self):
        """eps < 0 favours like neighbours, eps > 0 favours unlike ones."""
        blocky = BondEnergies.symmetric(-2.0, -1.0, -3.0)         # eps = -3
        alternating = BondEnergies.symmetric(-1.0, -3.0, -1.0)    # eps = +4
        assert blocky.nonadditivity < 0 < alternating.nonadditivity
        # Read off the ensemble rather than the parameters: at fixed length, which of the
        # two length-4 extremes carries more weight.
        for be, heavier, lighter in [(blocky, "0000", "0101"),
                                     (alternating, "0101", "0000")]:
            w = {s: _mass_action_weight(s, be, monomer=1.0, dg_assoc=0.0, rt=1.0)
                 for s in ("0000", "0101")}
            assert w[heavier] > w[lighter]

    def test_the_correlation_length_ignores_dg_assoc_and_an_even_monomer_supply(self):
        """Both scale the transfer matrix uniformly, so they cancel from the eigenvalue ratio.

        A version that folded `dg_assoc` in asymmetrically would still produce a plausible
        number here, just a different one for every food concentration.
        """
        base = sequence_correlation_length(NONADDITIVE)
        assert sequence_correlation_length(NONADDITIVE, dg_assoc=7.0) == pytest.approx(base)
        assert sequence_correlation_length(NONADDITIVE, monomer=0.01) == pytest.approx(base)

    def test_an_uneven_monomer_supply_does_NOT_cancel(self):
        assert sequence_correlation_length(
            NONADDITIVE, monomer=[0.01, 1.0]) != pytest.approx(
                sequence_correlation_length(NONADDITIVE))


class TestTheTransferMatrixReproducesMassAction:
    """The layered check: brute force first, and only then the matrix."""

    @pytest.mark.parametrize("be", [UNIFORM, ADDITIVE, NONADDITIVE])
    @pytest.mark.parametrize("dg_assoc", [0.0, 1.5])
    @pytest.mark.parametrize("monomer", [1.0, 0.3, [0.2, 0.7]])
    def test_matrix_weight_equals_mass_action_weight(self, be, dg_assoc, monomer):
        t = transfer_matrix(be, monomer=monomer, dg_assoc=dg_assoc)
        m = np.broadcast_to(np.asarray(monomer, dtype=float), (2,))
        for s in _sequences(7):
            idx = [be.alphabet.index(c) for c in s]
            via_matrix = m[idx[0]] * float(np.prod([t[i, j] for i, j
                                                    in zip(idx, idx[1:])]))
            direct = _mass_action_weight(s, be, monomer=monomer, dg_assoc=dg_assoc, rt=1.0)
            assert via_matrix == pytest.approx(direct, rel=1e-12)


class TestFlory:
    """The module's analytic anchor: a geometric length distribution, exactly."""

    def _weights_by_length(self, be, *, monomer, dg_assoc, max_len):
        out = np.zeros(max_len + 1)
        for s in _sequences(max_len, be.alphabet):
            out[len(s)] += _mass_action_weight(s, be, monomer=monomer,
                                               dg_assoc=dg_assoc, rt=1.0)
        return out

    def test_the_elongation_ratio_matches_the_hand_computed_one(self):
        """Uniform bond energy `E`, monomer `m`: two extensions per residue, each worth
        `K = exp(-(E + dg_assoc)/RT)`, so `rho = 2 m K` and nothing else."""
        rho = elongation_ratio(UNIFORM, monomer=0.1, dg_assoc=0.5)
        assert rho == pytest.approx(2 * 0.1 * np.exp(-(-1.0 + 0.5)))

    @pytest.mark.parametrize("be", [UNIFORM, ADDITIVE])
    def test_the_length_distribution_is_geometric_FROM_THE_FIRST_BOND(self, be):
        """Exact term by term for an additive assignment — but starting at length 2.

        The transfer matrix is a statement about bonds, and a monomer has none, so the
        length-1 term is a boundary condition and not the first term of the series.
        """
        w = self._weights_by_length(be, monomer=0.15, dg_assoc=0.0, max_len=10)
        rho = elongation_ratio(be, monomer=0.15, dg_assoc=0.0)
        for L in range(2, 10):
            assert w[L + 1] / w[L] == pytest.approx(rho, rel=1e-12), f"length {L}"

    def test_the_monomer_is_a_boundary_TERM_and_only_uniformity_hides_it(self):
        """The step 1 -> 2 need not equal rho, and under `UNIFORM` it accidentally does.

        That accident is why the boundary term is easy to miss: the null case is the one
        case where it is invisible. Checked on an assignment that is additive — same zero
        correlation length — but not uniform.
        """
        first = lambda be: (self._weights_by_length(be, monomer=0.15, dg_assoc=0.0,
                                                    max_len=2)[2]
                            / self._weights_by_length(be, monomer=0.15, dg_assoc=0.0,
                                                       max_len=2)[1])
        assert first(UNIFORM) == pytest.approx(elongation_ratio(UNIFORM, monomer=0.15),
                                               rel=1e-12)
        assert first(ADDITIVE) != pytest.approx(elongation_ratio(ADDITIVE, monomer=0.15),
                                                rel=1e-3)

    def test_the_boundary_term_costs_mean_length_1_4_percent_here(self):
        """`mean_length` returns the Flory value, which assumes the series starts at 1.

        Quantified rather than waved at, so that the docstring's "exact for a uniform
        assignment, asymptotic otherwise" has a number behind it.
        """
        be = ADDITIVE
        m = 0.4 / elongation_ratio(be, monomer=1.0)          # put rho at 0.4
        rho = elongation_ratio(be, monomer=m)
        w = self._weights_by_length(be, monomer=m, dg_assoc=0.0, max_len=16)
        n = w[1:].sum() + w[16] * rho / (1 - rho)
        lsum = (sum(L * w[L] for L in range(1, 17))
                + w[16] * rho * (17 - 16 * rho) / (1 - rho) ** 2)
        assert mean_length(be, monomer=m) / (lsum / n) == pytest.approx(1.014, abs=5e-4)

    def test_a_non_additive_assignment_is_geometric_only_ASYMPTOTICALLY(self):
        """And the approach is governed by the correlation length, which is why the two
        quantities are documented together."""
        w = self._weights_by_length(NONADDITIVE, monomer=0.02, dg_assoc=0.0, max_len=12)
        rho = elongation_ratio(NONADDITIVE, monomer=0.02, dg_assoc=0.0)
        early = abs(w[3] / w[2] - rho)
        late = abs(w[12] / w[11] - rho)
        assert late < early / 100.0

    def test_mean_length_is_one_over_one_minus_rho(self):
        """Checked against the enumerated distribution, with the truncated tail added back
        analytically so the comparison is not quietly a comparison to a truncated mean."""
        be, m = UNIFORM, 0.15
        rho = elongation_ratio(be, monomer=m, dg_assoc=0.0)
        w = self._weights_by_length(be, monomer=m, dg_assoc=0.0, max_len=12)
        # tail: w[L] = w[1] * rho**(L-1) for every L, so the omitted part is summable.
        n_head = w[1:].sum()
        lsum_head = sum(L * w[L] for L in range(1, 13))
        tail_n = w[1] * rho ** 12 / (1 - rho)
        tail_l = w[1] * rho ** 12 * (13 - 12 * rho) / (1 - rho) ** 2
        assert (lsum_head + tail_l) / (n_head + tail_n) == pytest.approx(
            mean_length(be, monomer=m), rel=1e-9)

    def test_runaway_polymerisation_raises_instead_of_returning_a_length(self):
        """`rho >= 1` has no equilibrium: any mean length reported there is a statement
        about `max_len`, not about the chemistry."""
        assert elongation_ratio(UNIFORM, monomer=2.0) > 1.0
        with pytest.raises(ValueError, match="runs away"):
            mean_length(UNIFORM, monomer=2.0)

    def test_dg_assoc_is_what_decides_whether_polymers_form_at_all(self):
        """It is the only sequence- and length-independent term, and the default of 0.0 is a
        choice — 'joining is free at the standard state' — not a neutral absence."""
        assert elongation_ratio(UNIFORM, monomer=0.5, dg_assoc=0.0) > 1.0
        assert elongation_ratio(UNIFORM, monomer=0.5, dg_assoc=3.0) < 1.0


class TestSignConvention:
    """`e` is a change, not a strength — the trap that produces plausible numbers."""

    def test_a_negative_bond_energy_favours_the_polymer(self):
        assert equilibrium_constant(BondEnergies.uniform(-2.0).ligation("0", "0")) > 1.0

    def test_a_POSITIVE_one_takes_it_apart(self):
        """Passing a bond 'strength' of +2 gives a chemistry that depolymerises, at a
        perfectly reasonable-looking rate."""
        assert equilibrium_constant(BondEnergies.uniform(2.0).ligation("0", "0")) < 1.0
        assert elongation_ratio(BondEnergies.uniform(2.0), monomer=1.0) < 1.0


class TestGuards:
    """Rejections, each of which was reachable by a plausible call."""

    def test_a_non_square_matrix_is_refused(self):
        with pytest.raises(ValueError, match="square"):
            BondEnergies(np.zeros((2, 3)))

    def test_alphabet_and_matrix_must_agree(self):
        with pytest.raises(ValueError, match="residues"):
            BondEnergies(np.zeros((2, 2)), alphabet="ACGT")

    def test_a_repeated_residue_is_refused(self):
        with pytest.raises(ValueError, match="repeated"):
            BondEnergies(np.zeros((2, 2)), alphabet="00")

    def test_an_unknown_residue_names_the_alphabet(self):
        with pytest.raises(ValueError, match="not in alphabet"):
            NONADDITIVE.sequence("012")

    def test_an_infinite_bond_energy_is_refused(self):
        with pytest.raises(ValueError, match="finite"):
            BondEnergies(np.array([[0.0, np.inf], [np.inf, 0.0]]))

    def test_ligating_an_empty_molecule_is_refused(self):
        with pytest.raises(ValueError, match="non-empty"):
            NONADDITIVE.ligation("01", "")

    def test_a_zero_enhancement_is_refused(self):
        with pytest.raises(ValueError, match="enablement"):
            catalysed_rates(rate_constants(0.0, barrier=1.0), 0.0)

    def test_a_four_letter_alphabet_works_and_has_no_scalar_epsilon(self):
        """The general form is here so a nucleotide chemistry needs no redesign; the
        binary shortcut is not, because for k > 2 the obstruction is a rank condition."""
        dna = BondEnergies(np.full((4, 4), -1.0), alphabet="ACGT")
        assert dna.sequence("ACGT") == pytest.approx(-3.0)
        assert sequence_correlation_length(dna) == 0.0
        with pytest.raises(ValueError, match="binary alphabet"):
            dna.nonadditivity

    def test_bond_energies_compare_by_identity_not_by_a_raising_elementwise_eq(self):
        """`eq=False` on the dataclass: the generated `__eq__` would call `bool()` on an
        array comparison and raise instead of answering."""
        assert BondEnergies.uniform(-1.0) != BondEnergies.uniform(-1.0)
        assert UNIFORM == UNIFORM
