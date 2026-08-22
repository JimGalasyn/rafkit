"""Thermodynamic consistency of autocatalytic cycles — is a set of cores realisable at once?

Implements the feasibility question of

    Kosc, Kuperberg, Rajon & Charlat, "Thermodynamic consistency of autocatalytic cycles",
    **PNAS 122(18) e2421274122 (2025)**, doi:10.1073/pnas.2421274122

⚠ **Like `dilution`, `permeation` and `thermo`, this is not a RAF algorithm.** It takes a
stoichiometric matrix and asks a question RAF theory cannot pose: *given that these reactions
must run in these directions, does any assignment of chemical potentials make that happen?*

**The reduction, and it is the whole module.** With `x_k = e^{μ_k}` the exponential of species
`k`'s chemical potential and `b_i = e^{−G‡_i}` the barrier factor, mass action under local
detailed balance gives::

    v_i = b_i · ( ∏_k x_k^{S⁻_{k,i}}  −  ∏_k x_k^{S⁺_{k,i}} )

`b_i > 0` always, so **the barrier scales the flow but cannot flip its sign**. Taking `y = ln x`
(which is `μ` itself), reaction `i` runs forward exactly when

    (Sᵀ y)_i < 0        i.e. affinity  A_i = −(Sᵀ y)_i > 0

so the question *"can all these reactions run forward at once?"* is **strict linear feasibility
of `Sᵀ y < 0`** — a linear program in the chemical potentials, and nothing else.

⚠⚠ **Therefore the verdict is independent of every rate constant and every barrier.** The paper
says so in as many words: *"this proof is valid regardless of the activation barrier values, since
the contradiction stems from the signs of the flows, while activation barriers only affect their
amplitudes."* Nothing in this module takes a rate constant, and that is not an omission.

**Two independent methods, required to agree.** Feasibility is decided by an LP *and* by Gordan's
theorem — exactly one of `Sᵀ y < 0` (a **witness**) and `S w = 0, w ≥ 0, w ≠ 0` (a **certificate**)
can hold. Disagreement raises. One view of a fact is an opinion; the module was built after a week
in which several checks reported something true about the wrong thing.

**Why this is the project's calibration anchor.** Kosc's **Theorem 2** guarantees a *single* PAC is
always consistent, so any network we generate supplies a pass/fail case with an external authority
deciding — and `tests/test_autocatalysis.py` carries their **Fig. 4** as a known answer.

⚠ **Their Fig. 3 is deliberately NOT here.** It is a pair of PACs that fail at the *flow* level —
contradictory requirements `v1 > v2` and `v1 < v2` — so it is not even a multiPAC, and the
obstruction never reaches thermodynamics. **This module implements the CAC layer only**; the
topological multiPAC test is a different feasibility problem and is not written.

⚠ `scipy` is an **optional** dependency (`pip install rafkit[cac]`), imported lazily. The rest of
rafkit stays numpy-only, which is the library's stated character.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Consistency", "is_thermodynamically_consistent", "stoichiometry", "affinities"]


@dataclass(frozen=True, eq=False)
class Consistency:
    """Verdict, with whichever of the two possible proofs applies.

    Exactly one of `witness` and `certificate` is not None — Gordan's theorem says the two
    alternatives are mutually exclusive and exhaustive, and `is_thermodynamically_consistent`
    checks that rather than trusting it.
    """

    consistent: bool
    witness: np.ndarray | None = None
    """Chemical potentials `μ = ln x` making every reaction run forward. `exp(witness)` is the
    `x` of the paper's Box 2. Only when `consistent`."""
    certificate: np.ndarray | None = None
    """Non-negative `w ≠ 0` with `S w = 0`: a non-negative flow combination that cancels, which
    is why no potential assignment can drive them all forward. Only when NOT `consistent`."""
    margin: float = 0.0
    """How strictly the witness satisfies `Sᵀy < 0` — `min_i (−Sᵀy)_i`. Zero when infeasible."""

    def __bool__(self) -> bool:
        return self.consistent


def stoichiometry(reactions, species=None):
    """Build `S` (species × reactions) from `(reactants, products)` pairs of species names.

    Each reaction is a pair of sequences; repeats mean stoichiometric coefficients, so
    `(["e3"], ["e4", "e4"])` is `e3 → 2 e4`. Returns `(S, species)`.
    """
    # Materialised once: this is iterated up to three times below, so a generator used to
    # raise `TypeError` on len(), and a single-pass iterable that *did* report a length would
    # have come back exhausted and produced an all-zero S -- a silently wrong matrix.
    reactions = [(list(lhs), list(rhs)) for lhs, rhs in reactions]
    if species is None:
        seen = {}
        for lhs, rhs in reactions:
            for x in lhs + rhs:
                seen.setdefault(x, None)
        species = list(seen)
    species = list(species)
    idx = {x: i for i, x in enumerate(species)}
    unknown = {x for lhs, rhs in reactions for x in lhs + rhs} - set(idx)
    if unknown:
        # A bare KeyError names the missing species but not the reason; the caller's mistake
        # is an incomplete `species` list, and the message should say so.
        raise ValueError(f"reaction(s) reference {sorted(unknown)}, which are not in the given "
                         f"species list {species}. Pass species=None to discover them.")
    S = np.zeros((len(species), len(reactions)))
    for j, (lhs, rhs) in enumerate(reactions):
        for x in lhs:
            S[idx[x], j] -= 1.0
        for x in rhs:
            S[idx[x], j] += 1.0
    return S, species


def affinities(S, y):
    """`A = −Sᵀy`, the affinity of each reaction. Positive means it runs in its stored direction."""
    return -(np.asarray(S, dtype=float).T @ np.asarray(y, dtype=float))


def _normalise_columns(S):
    """Scale each reaction column to unit norm, and report the zero columns.

    ⚠ **Exact, not an approximation.** Both Gordan alternatives are preserved by *positive*
    column scaling: `Sᵀy < 0` scales inequality `i` by `c_i > 0`, and `S w = 0, w ≥ 0` maps to
    `(SD)(D⁻¹w) = 0` with `D⁻¹w` still non-negative and non-zero. So normalising changes no
    verdict while making the LPs scale-free.

    Returns `(S_normalised, zero_column_indices, norms)`; `norms` maps a certificate back into
    the caller's coordinates.

    Without it the module returned a **wrong verdict with an invalid certificate** on an
    ill-conditioned input: `A → B` scaled by `1e-10` came back `consistent=False` with
    `w = [1.0]`, for which `S w = 1e-10·[−1,1] ≠ 0`. Both LPs were fooled the same way, so the
    cross-check stayed silent — tuning tolerances would have moved the cliff, not removed it.
    """
    S = np.asarray(S, dtype=float)
    norms = np.linalg.norm(S, axis=0)
    zero = np.flatnonzero(norms == 0.0)
    safe = np.where(norms > 0.0, norms, 1.0)
    return S / safe, zero, safe


def _gordan_certificate(S, tol=1e-9):
    """Non-negative `w` with `Σw = 1` and `S w = 0`, or None. `S` must be column-normalised.

    Gordan's theorem: exactly one of `Sᵀy < 0` and `{S w = 0, w ≥ 0, w ≠ 0}` is solvable.
    Posed as a **bounded feasibility** program — `Σw = 1` fixes the scale a cone would otherwise
    leave free, which is far better conditioned than maximising `Σw` under a box.

    ⚠ **The returned `w` is verified, not trusted.** `res.success` is the solver's opinion about
    its own convergence; `‖S w‖∞ ≤ tol` is the property the caller needs. Those are different
    claims, and the first used to be accepted for the second.
    """
    from scipy.optimize import linprog

    S = np.asarray(S, dtype=float)
    r = S.shape[1]
    res = linprog(c=np.zeros(r),
                  A_eq=np.vstack([S, np.ones(r)]),
                  b_eq=np.concatenate([np.zeros(S.shape[0]), [1.0]]),
                  bounds=[(0.0, None)] * r, method="highs")
    if not res.success or res.x is None:
        return None
    w = np.asarray(res.x)
    if w.min() < -tol or abs(w.sum() - 1.0) > 1e-6:
        return None
    w = np.clip(w, 0.0, None)
    return w if float(np.abs(S @ w).max()) <= tol else None


def is_thermodynamically_consistent(S, *, tol: float = 1e-9) -> Consistency:
    """Can every reaction of `S` run forward at once, for some assignment of chemical potentials?

    `S` is species × reactions, **with each column already oriented in the direction that
    reaction must run**. The answer is Kosc et al.'s CAC question: a *PAC* is a topological
    claim about stoichiometry, a *CAC* additionally requires a point in concentration space that
    realises it, and this decides the second.

    Returns a `Consistency` carrying the proof: a `witness` `μ` when feasible, a Gordan
    `certificate` `w` when not.

    ⚠ **Takes no rate constants and no barriers, deliberately** — see the module docstring. A
    caller who expects to pass kinetics is misreading the result, not missing an argument.
    """
    S_raw = np.asarray(S, dtype=float)
    if S_raw.ndim != 2 or S_raw.size == 0:
        raise ValueError(f"S must be a non-empty species x reactions matrix, got shape "
                         f"{S_raw.shape}")
    from scipy.optimize import linprog

    S, zero_cols, norms = _normalise_columns(S_raw)
    if zero_cols.size:
        # A reaction with no net stoichiometry has (Sᵀy)_i = 0 for every y, which is never < 0.
        # It is its own certificate: S e_i = 0 with e_i ≥ 0, non-zero.
        w = np.zeros(S.shape[1])
        w[zero_cols[0]] = 1.0
        return Consistency(False, certificate=w)

    s, r = S.shape
    # max t  s.t.  Sᵀy + t ≤ 0,  t ≤ 1,  y boxed to keep the LP finite. Feasibility is a cone,
    # so any solution can be scaled to margin 1; the box only has to be wide enough to reach it.
    A_ub = np.hstack([S.T, np.ones((r, 1))])
    res = linprog(c=np.concatenate([np.zeros(s), [-1.0]]), A_ub=A_ub, b_ub=np.zeros(r),
                  bounds=[(-1e6, 1e6)] * s + [(None, 1.0)], method="highs")
    # Verified, not trusted: the witness must actually satisfy Sᵀy < 0.
    feasible = False
    if res.success and res.x is not None and res.x[-1] > tol:
        feasible = bool(np.max(S.T @ res.x[:s]) < 0.0)

    cert = _gordan_certificate(S, tol=tol)
    if cert is not None:
        # Back into the CALLER's coordinates. `S_norm = S_raw/‖·‖`, so `S_norm w = S_raw (w/‖·‖)`
        # -- without this the returned certificate solves a matrix the caller never passed, and
        # `S_raw @ certificate` is not zero. Caught by the Fig. 4 null-cycle test.
        cert = cert / norms
    # Gordan: exactly one alternative holds. Asserted, not assumed -- two views of one fact.
    if feasible == (cert is not None):
        raise RuntimeError(
            f"Gordan's alternative violated: LP says feasible={feasible} while a non-negative "
            f"null combination was {'found' if cert is not None else 'not found'}. One of the "
            "two computations is wrong; the verdict is not usable.")
    if feasible:
        y = np.asarray(res.x[:s])
        return Consistency(True, witness=y, margin=float(res.x[-1]))
    return Consistency(False, certificate=cert)
