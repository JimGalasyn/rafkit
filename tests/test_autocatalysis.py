"""Thermodynamic consistency of autocatalytic cycles — against Kosc et al.'s published answers.

This is the project's **external calibration anchor**, and the reason it is worth more than the
`f ≈ 1.20` figure it replaces: those were a published *figure* matched with a *fitted* parameter,
while these are exact verdicts with no free parameter at all.

    Kosc, Kuperberg, Rajon & Charlat, "Thermodynamic consistency of autocatalytic cycles",
    PNAS 122(18) e2421274122 (2025).

⚠ The Fig. 4 network below is **reconstructed**, not copied: the paper draws it as a figure. It
comes from Box 2's flow equations (`v1 = b1(x1 − x_B x2)` etc.) and was cross-checked against the
figure's composition glyphs — every reaction mass-balances in (white, grey) units, both cores are
autocatalytic in `e4`, and they share exactly `{R2, R3}`, matching the caption's "two reactions in
common". Recorded as reconstructed so that a future mismatch is debugged here first.
"""
from __future__ import annotations

import numpy as np
import pytest

from rafkit.autocatalysis import (Consistency, affinities,
                                  is_thermodynamically_consistent, stoichiometry)

# --- Kosc et al. Fig. 4 -------------------------------------------------------------
RX = {
    "R1":  (["e1"],         ["eB", "e2"]),
    "R2":  (["e2", "e2p"],  ["e3"]),          # shared between the two cores
    "R3":  (["e3"],         ["e4", "e4"]),    # shared between the two cores
    "R4":  (["eA", "e4"],   ["e1"]),
    "R1p": (["e1p"],        ["eA", "e2p"]),
    "R4p": (["eB", "e4"],   ["e1p"]),
}
SPECIES = ["eA", "eB", "e1", "e2", "e3", "e4", "e1p", "e2p"]
PAC1 = ["R1", "R2", "R3", "R4"]
PAC2 = ["R1p", "R2", "R3", "R4p"]
BOTH = ["R1", "R2", "R3", "R4", "R1p", "R4p"]


def S_of(names):
    return stoichiometry([RX[n] for n in names], SPECIES)[0]


class TestKoscFigure4:
    """The published verdict: a multiPAC that is not a multiCAC."""

    def test_the_reconstruction_mass_balances(self):
        """(white, grey) composition per the figure's circle glyphs. If this fails, the network
        is wrong and every verdict below is about a different chemistry."""
        comp = {"eA": (1, 0), "eB": (0, 1), "e2": (2, 0), "e2p": (0, 2),
                "e4": (1, 1), "e1": (2, 1), "e1p": (1, 2), "e3": (2, 2)}
        for name, (lhs, rhs) in RX.items():
            L = tuple(sum(comp[s][i] for s in lhs) for i in (0, 1))
            R = tuple(sum(comp[s][i] for s in rhs) for i in (0, 1))
            assert L == R, name

    def test_the_two_cores_share_exactly_two_reactions(self):
        assert sorted(set(PAC1) & set(PAC2)) == ["R2", "R3"]

    @pytest.mark.parametrize("core", [PAC1, PAC2], ids=["PAC1", "PAC2"])
    def test_each_core_ALONE_is_consistent(self, core):
        """Kosc **Theorem 2**: a single PAC is always thermodynamically consistent.

        This is the anchor's load-bearing half — an *external* guarantee our implementation can
        violate. If a lone PAC ever comes back inconsistent, the implementation is wrong.
        """
        r = is_thermodynamically_consistent(S_of(core))
        assert r.consistent
        assert np.all(affinities(S_of(core), r.witness) > 0)
        assert r.certificate is None and r.margin > 0

    def test_TOGETHER_they_are_NOT_consistent(self):
        """Box 2's result: topologically compatible, thermodynamically not."""
        r = is_thermodynamically_consistent(S_of(BOTH))
        assert not r.consistent
        assert r.witness is None and r.certificate is not None

    def test_and_the_obstruction_is_an_exact_null_cycle(self):
        """*Why* it fails, checked rather than accepted.

        The certificate comes back with every weight equal: the six reactions at unit flux
        return the system to its starting composition. Then `Σ w_i A_i = −(S w)·y = 0`, so the
        affinities cannot all be positive — **a cycle that returns to its starting composition
        cannot be downhill all the way round.**
        """
        S = S_of(BOTH)
        w = is_thermodynamically_consistent(S).certificate
        assert np.all(w > 0)                                  # every reaction participates
        assert np.allclose(w / w.max(), 1.0)                  # ...at equal flux
        assert np.allclose(S @ w, 0.0, atol=1e-12)            # ...and it cancels exactly


class TestTheReductionItself:
    """Small cases where the answer is obvious, so the machinery can be wrong visibly."""

    def test_a_lone_irreversible_reaction_is_consistent(self):
        S, _ = stoichiometry([(["A"], ["B"])])
        assert is_thermodynamically_consistent(S)

    def test_a_reaction_and_its_reverse_cannot_both_run(self):
        """The smallest possible null cycle, and the smallest possible second-law violation."""
        S, _ = stoichiometry([(["A"], ["B"]), (["B"], ["A"])])
        r = is_thermodynamically_consistent(S)
        assert not r.consistent
        assert np.allclose(S @ r.certificate, 0.0)

    def test_a_three_step_loop_cannot_be_downhill_all_the_way_round(self):
        S, _ = stoichiometry([(["A"], ["B"]), (["B"], ["C"]), (["C"], ["A"])])
        assert not is_thermodynamically_consistent(S)

    def test_a_three_step_chain_can(self):
        S, _ = stoichiometry([(["A"], ["B"]), (["B"], ["C"]), (["C"], ["D"])])
        assert is_thermodynamically_consistent(S)

    def test_the_witness_really_orders_the_potentials(self):
        """`A → B → C` downhill means `μ_A > μ_B > μ_C`. Checked, not assumed."""
        S, sp = stoichiometry([(["A"], ["B"]), (["B"], ["C"])])
        y = is_thermodynamically_consistent(S).witness
        mu = dict(zip(sp, y))
        assert mu["A"] > mu["B"] > mu["C"]

    def test_barriers_and_rate_constants_are_ABSENT_by_design(self):
        """The verdict cannot depend on kinetics, so the API cannot accept any.

        Kosc: *"valid regardless of the activation barrier values, since the contradiction stems
        from the signs of the flows, while activation barriers only affect their amplitudes."*
        """
        import inspect
        params = set(inspect.signature(is_thermodynamically_consistent).parameters)
        assert not (params & {"k", "kinetics", "barrier", "rates", "kappa", "b"})
        assert params == {"S", "tol"}


class TestGordanAlternativeHolds:
    """The two methods must always agree; the module raises if they do not."""

    def test_on_many_random_networks(self):
        """Exercises the internal cross-check. A raise here means one method is wrong."""
        rng = np.random.default_rng(0)
        n_feas = 0
        for _ in range(120):
            s, r = rng.integers(2, 7), rng.integers(2, 7)
            S = rng.integers(-2, 3, size=(s, r)).astype(float)
            res = is_thermodynamically_consistent(S)     # raises if Gordan is violated
            assert (res.witness is None) != (res.certificate is None)
            n_feas += res.consistent
        assert 0 < n_feas < 120, "degenerate sample: need both verdicts represented"

    def test_a_null_combination_always_forbids_consistency(self):
        """If some strictly positive `w` has `S w = 0`, no potential assignment can work."""
        rng = np.random.default_rng(3)
        for _ in range(25):
            S = rng.integers(-2, 3, size=(4, 3)).astype(float)
            S = np.hstack([S, -S.sum(axis=1, keepdims=True)])   # force a positive null vector
            assert np.allclose(S @ np.ones(S.shape[1]), 0.0)
            assert not is_thermodynamically_consistent(S).consistent


class TestGuards:

    def test_an_empty_matrix_is_refused(self):
        with pytest.raises(ValueError, match="non-empty"):
            is_thermodynamically_consistent(np.zeros((0, 0)))

    def test_a_one_dimensional_input_is_refused(self):
        with pytest.raises(ValueError, match="species x reactions"):
            is_thermodynamically_consistent(np.array([1.0, -1.0]))

    def test_stoichiometry_counts_repeats_as_coefficients(self):
        S, sp = stoichiometry([(["e3"], ["e4", "e4"])], ["e3", "e4"])
        assert S[sp.index("e4"), 0] == 2.0 and S[sp.index("e3"), 0] == -1.0

    def test_consistency_is_truthy(self):
        assert bool(Consistency(True)) and not bool(Consistency(False))


class TestTheCrossCheckCanActuallyFire:
    """A guard that has never failed has not been shown to be able to.

    The module's two-method design is only worth something if disagreement is detected. These
    force a disagreement and a solver failure, which no honest input reaches.
    """

    def test_disagreement_between_the_two_methods_RAISES(self, monkeypatch):
        """Make the certificate lie, and the LP must catch it.

        `A → B` is plainly consistent, so a certificate claiming otherwise is a contradiction.
        Without this test the cross-check could be dead code and the suite would stay green.
        """
        import rafkit.autocatalysis as ac
        monkeypatch.setattr(ac, "_gordan_certificate", lambda S, tol=1e-9: np.ones(S.shape[1]))
        S, _ = stoichiometry([(["A"], ["B"])])
        with pytest.raises(RuntimeError, match="Gordan's alternative violated"):
            ac.is_thermodynamically_consistent(S)

    def test_and_in_the_other_direction_too(self, monkeypatch):
        """A null cycle IS inconsistent, so a certificate of None is the opposite lie."""
        import rafkit.autocatalysis as ac
        monkeypatch.setattr(ac, "_gordan_certificate", lambda S, tol=1e-9: None)
        S, _ = stoichiometry([(["A"], ["B"]), (["B"], ["A"])])
        with pytest.raises(RuntimeError, match="Gordan's alternative violated"):
            ac.is_thermodynamically_consistent(S)

    def test_a_failed_solver_yields_no_certificate_rather_than_a_wrong_one(self, monkeypatch):
        """If the LP cannot answer, `_gordan_certificate` must return None, not guess."""
        import rafkit.autocatalysis as ac
        import scipy.optimize as so

        class Failed:
            success, x = False, None
        monkeypatch.setattr(so, "linprog", lambda *a, **k: Failed())
        assert ac._gordan_certificate(np.array([[1.0, -1.0]])) is None
