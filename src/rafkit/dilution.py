"""Serial-dilution and CSTR protocols for competing autocatalytic sets.

⚠ **This module is deliberately NOT about RAF sets, and that is the point.** Everything
else in rafkit takes a `ReactionNetwork` and asks a RAF question of it. This module takes
no network at all: it is a two-species ordinary differential equation, reproducing the
minimal model of

    Matsubara, Ameta, Thutupalli, Nghe & Krishna, "Conditions for Darwinian evolution in
    compartmentalized autocatalytic reaction networks", arXiv:2211.03155.

It is here because it is the library's only **analytic** calibration target. Every other
check in rafkit is against a reference implementation (CatReNet) or a published *figure*
(Steel, Hordijk & Smith); those can only be matched qualitatively or to a count. Matsubara
et al. derive closed-form conditions, which can be hit or missed to nine significant
figures — and the four checks in `tests/test_dilution.py` do hit them.

The wider use is that RAF work increasingly runs networks inside growing, dividing
compartments, where the dilution protocol is a modelling choice that changes the answer.
This module gives that choice a validated implementation and a benchmark, independent of
any RAF structure.

**The model.** Two identical but distinguishable autocatalytic species competing for one
substrate::

    dx_i/dt = s * r(x_i) * x_i          i = 1, 2
    s       = s_tot - x_1 - x_2         (replenishment compensates dilution)

with `r(x) x` the reproduction flux; they study `r(x)x = eps + kappa x^2`.

**The protocols.** Serial dilution (SD) integrates for `dt` then divides everything by
`m = exp(phi * dt)`; CSTR applies the same average dilution continuously and is the
`dt -> 0` limit of SD. General growth-division protocols interpolate between the two.

**Their claims, all analytic and all checked in the tests:**

1. `r(x)x` LINEAR (`eps + kappa x`) => only the symmetric trajectory is stable: **no
   bistability**, hence no compositional heredity.
2. `r(x)x = eps + kappa x^2` => **bistability**, at their `dt = 1`, `kappa = 8`,
   `eps = 0.5`, `phi = 1`.
3. Bistability is **lost above a critical cycle interval** `dt`.
4. Their equation (2), parameter-free::

       delta(t)/chi(t) = [ r(chi(t)/2) / r(chi(0)/2) ] * delta(0)/chi(0)

   with `chi = x1 + x2`, `delta = x1 - x2`, near the symmetric trajectory. The
   amplification factor crossing 1 is their equation (3), the sufficient condition for
   bistability under SD.

⚠ `s_tot` is not stated in their text (it is `sigma/phi` for an unstated `sigma`), so it is
a parameter here. Claims 1, 3 and 4 do not depend on it.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np


def flux_linear(x, eps: float = 0.5, kappa: float = 8.0):
    """`r(x) x = eps + kappa x` — their NO-bistability case."""
    return eps + kappa * np.asarray(x, dtype=float)


def flux_quadratic(x, eps: float = 0.5, kappa: float = 8.0):
    """`r(x) x = eps + kappa x^2` — their bistable case (equation 4)."""
    x = np.asarray(x, dtype=float)
    return eps + kappa * x * x


_TINY = 1e-150


def _rate(flux, x):
    """`r(x)` itself, i.e. flux divided by x.

    The `x -> 0` limit is taken FROM THE FLUX rather than hardcoded. An earlier version
    returned `inf` there, which is right for `eps + kappa x^2` with `eps > 0` but wrong
    whenever the flux vanishes at the origin -- `eps = 0` gives `r(x) = kappa x`, whose
    limit is 0, not infinity.
    """
    x = np.asarray(x, dtype=float)
    safe = np.where(x > _TINY, x, _TINY)
    return np.asarray(flux(safe), dtype=float) / safe


def _heun_step(x, h, deriv):
    """One Heun step. Shared so the two protocols cannot drift apart."""
    d1 = deriv(x)
    xp = np.maximum(x + h * d1, 0.0)
    return np.maximum(x + h * 0.5 * (d1 + deriv(xp)), 0.0)


@dataclass
class DilutionResult:
    """The composition after a run.

    Deliberately carries NO `bistable` field. Bistability is a property of a PAIR of runs
    from different initial conditions -- a single trajectory cannot know it -- and an
    earlier version had exactly that field, permanently `False`, which is worse than
    absent because it reads like an answer. Use `is_bistable`.
    """

    x: np.ndarray
    cycles: int


def run_serial_dilution(x0, *, flux=flux_quadratic, dt: float = 1.0, phi: float = 1.0,
                        s_tot: float = 1.0, cycles: int = 400,
                        steps_per_cycle: int = 2000) -> DilutionResult:
    """Integrate `cycles` serial-dilution cycles and return the final composition.

    Each cycle integrates for `dt` then divides every species by `m = exp(phi*dt)`, which
    is their SD protocol exactly. Taking `dt -> 0` at fixed `phi` gives the CSTR limit,
    available through `run_cstr`.

    `steps_per_cycle` is set from a measured convergence sweep, not guessed: Heun is
    second order and the composition fraction settles as 7.2e-6 / 1.8e-6 / 4.5e-7 / 1.1e-7
    at 500 / 1000 / 2000 / 4000 steps. 2000 is two orders tighter than the 1e-3 tolerance a
    bistability verdict uses, at half the cost of the previous default. Raise it for work
    that needs the analytic equation to nine figures -- `tests/test_dilution.py` uses
    20 000 there.
    """
    x = np.array(x0, dtype=float)
    m = float(np.exp(phi * dt))
    h = dt / steps_per_cycle

    def deriv(y):                      # no dilution term: SD dilutes discretely, below
        return max(s_tot - y.sum(), 0.0) * flux(y)

    for _ in range(cycles):
        for _ in range(steps_per_cycle):
            x = _heun_step(x, h, deriv)
        x = x / m
    return DilutionResult(x=x, cycles=cycles)


def run_cstr(x0, *, flux=flux_quadratic, phi: float = 1.0, s_tot: float = 1.0,
             t_end: float = 400.0, steps: int = 400_000) -> DilutionResult:
    """The CSTR limit: the same average dilution applied continuously."""
    x = np.array(x0, dtype=float)
    h = t_end / steps

    def deriv(y):                      # dilution is continuous here, not a discrete cut
        return max(s_tot - y.sum(), 0.0) * flux(y) - phi * y

    for _ in range(steps):
        x = _heun_step(x, h, deriv)
    return DilutionResult(x=x, cycles=0)


def is_bistable(runner, *, asym: float = 0.2, tol: float = 1e-3, s_tot: float = 1.0,
                **kw) -> bool:
    """Does the system remember which species started ahead?

    Their criterion: with two symmetric species, a system WITHOUT heredity settles on the
    symmetric trajectory `x1 = x2` from every initial condition, while a bistable one keeps
    whichever started dominant. Run both asymmetries and require the outcomes to differ.

    The two species start summing to `s_tot / 2`, so a caller who changes `s_tot` gets an
    initial condition that still scales with it. An earlier version hardcoded that total at
    0.5 while forwarding `s_tot` onward, which silently mismatched the two.

    `**kw` is filtered to the parameters `runner` actually accepts, so the same call works
    for both protocols -- `run_cstr` takes no `dt`, and forwarding one used to raise
    `TypeError` on documented usage. Filtering is safe here because these are *protocol*
    parameters: a cycle interval is meaningless for a CSTR, which has no cycles.
    """
    accepted = set(inspect.signature(runner).parameters)
    passed = {k: v for k, v in kw.items() if k in accepted}
    if "s_tot" in accepted:
        passed["s_tot"] = s_tot

    half = s_tot / 4.0                      # each species; the pair sums to s_tot / 2
    a = runner([half * (1 + asym), half * (1 - asym)], **passed).x
    b = runner([half * (1 - asym), half * (1 + asym)], **passed).x
    if a.sum() <= 0 or b.sum() <= 0:
        return False
    fa = a[0] / a.sum()
    fb = b[0] / b.sum()
    # A PYTHON bool, deliberately. Returning np.bool_ makes `result is False` silently
    # never match, which is not a hypothetical: it made a bifurcation sweep report "no
    # transition found" over data that plainly showed one.
    return bool(abs(fa - fb) > tol)


def symmetric_growth_ratio(flux, chi_0: float, chi_t: float) -> float:
    """Their equation (2)'s amplification factor `r(chi_t/2) / r(chi_0/2)`.

    Greater than 1 means the symmetric trajectory is unstable over the cycle, which is
    their **sufficient condition for bistability under SD** (equation 3).
    """
    return float(_rate(flux, np.array([chi_t / 2.0]))[0]
                 / _rate(flux, np.array([chi_0 / 2.0]))[0])
