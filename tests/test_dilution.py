"""Calibration against Matsubara et al. (arXiv:2211.03155) -- four analytic claims.

rafkit's only ANALYTIC calibration. Every other check in this library is against a
reference implementation (CatReNet) or a published figure (Steel, Hordijk & Smith), which
can be matched to a count or a shape. These are closed-form conditions, so they can be hit
or missed to nine significant figures -- and a wrong integrator would miss them.

Their model: dx_i/dt = s r(x_i) x_i for two symmetric competing autocatalytic species, with
substrate replenishment holding s_tot fixed, under serial dilution.
"""
from __future__ import annotations

import numpy as np
import pytest

from rafkit.dilution import (flux_linear, flux_quadratic, is_bistable,
                                  run_cstr, run_serial_dilution,
                                  symmetric_growth_ratio)

# Their stated parameters (Fig. 3): dt = 1, kappa = 8, eps = 0.5, phi = 1.
KW = dict(dt=1.0, phi=1.0, cycles=200)


class TestTheirAnalyticClaims:
    def test_linear_flux_gives_NO_bistability(self):
        """Their claim: with r(x)x linear, "only the growth state with delta = 0 (i.e. the
        symmetrical trajectory) is always stable" -- so no compositional heredity. This is
        the negative half of their section heading, "heredity under serial dilution requires
        a concentration-dependent growth rate"."""
        assert not is_bistable(run_serial_dilution, flux=flux_linear, **KW)

    def test_quadratic_flux_gives_bistability(self):
        """Their equation (4) case, at their stated parameters."""
        assert is_bistable(run_serial_dilution, flux=flux_quadratic, **KW)

    def test_bistability_is_lost_above_a_critical_cycle_interval(self):
        """Their Fig. 3B: a bifurcation at dt = dt_c, above which heredity is gone."""
        short = is_bistable(run_serial_dilution, flux=flux_quadratic, dt=0.5, phi=1.0,
                            cycles=150)
        long_ = is_bistable(run_serial_dilution, flux=flux_quadratic, dt=3.0, phi=1.0,
                            cycles=150)
        assert short and not long_

    @pytest.mark.parametrize("flux,expect_unstable",
                             [(flux_linear, False), (flux_quadratic, True)])
    def test_equation_2_holds(self, flux, expect_unstable):
        """Their equation (2), which is parameter-free:

            delta(t)/chi(t) = [r(chi(t)/2) / r(chi(0)/2)] * delta(0)/chi(0)

        Checked near the symmetric trajectory over one cycle with dilution off. The
        amplification factor crossing 1 is their equation (3), the sufficient condition for
        bistability -- so the two parametrisations must also AGREE on which case is
        unstable, which is the second assertion.
        """
        x0 = np.array([0.25 * (1 + 1e-6), 0.25 * (1 - 1e-6)])
        r = run_serial_dilution(x0, flux=flux, dt=1.0, phi=0.0, cycles=1,
                                steps_per_cycle=20_000)
        chi0, d0 = x0.sum(), x0[0] - x0[1]
        chit, dt_ = r.x.sum(), r.x[0] - r.x[1]
        measured = (dt_ / chit) / (d0 / chi0)
        predicted = symmetric_growth_ratio(flux, chi0, chit)
        assert measured == pytest.approx(predicted, rel=1e-6)
        assert (predicted > 1.0) is expect_unstable


class TestTheProtocolsRelateAsTheyShould:
    def test_short_cycles_approach_the_cstr_limit(self):
        """Their framing: CSTR is the dt -> 0 limit of serial dilution, and general
        growth-division protocols are bounded by the two."""
        a = run_serial_dilution([0.3, 0.2], flux=flux_quadratic, dt=0.02, phi=1.0,
                                cycles=4000, steps_per_cycle=200).x
        b = run_cstr([0.3, 0.2], flux=flux_quadratic, phi=1.0, t_end=80.0,
                     steps=80_000).x
        assert a.sum() == pytest.approx(b.sum(), rel=0.1), (a, b)


class TestTheApiDoesNotLie:
    def test_is_bistable_returns_a_python_bool(self):
        """Regression. It returned np.bool_, so `result is False` never matched and a
        bifurcation sweep reported "no transition found" over data that showed one."""
        got = is_bistable(run_serial_dilution, flux=flux_linear, dt=1.0, phi=1.0,
                          cycles=20)
        assert got is False or got is True
