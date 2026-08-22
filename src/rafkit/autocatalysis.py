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
deciding — and `tests/test_autocatalysis.py` carries their Fig. 3 and Fig. 4 as known answers.

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
    if species is None:
        seen = {}
        for lhs, rhs in reactions:
            for s in list(lhs) + list(rhs):
                seen.setdefault(s, None)
        species = list(seen)
    idx = {s: i for i, s in enumerate(species)}
    S = np.zeros((len(species), len(reactions)))
    for j, (lhs, rhs) in enumerate(reactions):
        for s in lhs:
            S[idx[s], j] -= 1.0
        for s in rhs:
            S[idx[s], j] += 1.0
    return S, list(species)


def affinities(S, y):
    """`A = −Sᵀy`, the affinity of each reaction. Positive means it runs in its stored direction."""
    return -(np.asarray(S, dtype=float).T @ np.asarray(y, dtype=float))


def _gordan_certificate(S, tol=1e-9):
    """Non-negative `w ≠ 0` with `S w = 0`, or None.

    Gordan's theorem: exactly one of `Sᵀy < 0` and `{S w = 0, w ≥ 0, w ≠ 0}` is solvable. Found
    by an LP maximising `Σw` under `S w = 0`, `0 ≤ w ≤ 1` — a bounded feasible program whose
    optimum is 0 precisely when no such `w` exists.
    """
    from scipy.optimize import linprog

    S = np.asarray(S, dtype=float)
    r = S.shape[1]
    res = linprog(c=-np.ones(r), A_eq=S, b_eq=np.zeros(S.shape[0]),
                  bounds=[(0.0, 1.0)] * r, method="highs")
    if not res.success or res.x is None:
        return None
    w = np.asarray(res.x)
    return w if w.sum() > tol else None


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
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.size == 0:
        raise ValueError(f"S must be a non-empty species x reactions matrix, got shape {S.shape}")
    from scipy.optimize import linprog

    s, r = S.shape
    # max t  s.t.  Sᵀy + t ≤ 0,  t ≤ 1,  y free (bounded to keep the LP finite).
    # A strictly positive optimum is a strict-feasibility witness; t is the margin.
    A_ub = np.hstack([S.T, np.ones((r, 1))])
    res = linprog(c=np.concatenate([np.zeros(s), [-1.0]]), A_ub=A_ub, b_ub=np.zeros(r),
                  bounds=[(-1e3, 1e3)] * s + [(None, 1.0)], method="highs")
    feasible = bool(res.success and res.x is not None and res.x[-1] > tol)

    cert = _gordan_certificate(S, tol=tol)
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
