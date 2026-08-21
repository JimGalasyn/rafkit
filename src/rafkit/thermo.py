"""Free energy for a polymer chemistry — bond energies, and the rate constants they force.

⚠ **Like `dilution` and `permeation`, this is not a RAF algorithm.** It is here because
the rest of the library already has a thermodynamics — an *implicit* one, never written
down — and it is the wrong one.

**What the library currently asserts without saying so.** `gillespie` gives every reaction
a unit rate constant. In a cleavage–ligation chemistry that means `k_f = k_r` for every
reaction, hence `K_eq = 1`, hence `ΔG° = 0`: **every polymer is isoenergetic with the parts
it is made of.** Nothing is more stable than anything else, no sequence is preferred over
any other, and the polymer growth those runs show is driven entirely by the food boundary
condition. That is not a badly chosen parameter. It is free energy being *absent* while the
model talks as though it were present.

**Catalysis has the same problem one level down.** Where the uncatalysed rate is zero,
"catalysis" is not acceleration but **enablement**: the catalyst decides whether the
reaction exists at all, so "which species catalyses what" is a choice of *which reactions
there are*, wearing a kinetic name. A catalyst that enables also moves the equilibrium —
from unreachable to reachable — and **no catalyst does that**. Here catalysis is a ratio
`k_cat/k_uncat` applied to **both directions at once**, which is the only form that leaves
`K_eq` alone; see `catalysed_rates`. For scale: 100× is a barrier drop of `ln(100)·RT` =
11.4 kJ/mol at 298 K, about one hydrogen bond, and sits at the bottom of the weakest
catalysis class (mineral 10¹–10⁴, ribozymes 10³–10⁹, enzymes 10⁶–10¹⁷).

⚠ rafkit's own `gillespie` runs an uncatalysed reaction at `1/20` rather than at zero, so it
already *has* a rate enhancement. But 20 there is a bare number: with no reverse rate derived
from a free energy there is nothing for it to be a ratio *of*. This module supplies the
missing half.

**Why three bond energies and not one.** Virgo's construction assigns a fixed free energy per
bond, so a polymer's free energy is proportional to its length. ⚠ **A uniform bond energy
gives every equal-length sequence identical free energy**, so no thermodynamic sequence
preference can exist. In a unary alphabet the question cannot arise; in a binary one it is the
whole point. `E₀₀ / E₀₁ / E₁₁` is the smallest assignment under which it can.

**And the sharp form of that is not "uniform".** The ensemble these energies induce is exactly
a one-dimensional Ising chain, and its transfer matrix has a second eigenvalue that vanishes
whenever

    ε = E₀₀ + E₁₁ − E₀₁ − E₁₀  =  0

— the **additive** case, where `E[a][b] = h(a) + g(b)` and the bond energy carries no
information about the *pair*. Every additive assignment has zero sequence correlation length,
uniform or not, so three energies that happen to satisfy `E₀₁ = (E₀₀+E₁₁)/2` buy nothing over
one. (Under `symmetric()`, where `E₀₁ = E₁₀`, `ε` is the design's `E₀₀ + E₁₁ − 2·E₀₁`; the
matrix is not required to be symmetric, and the doubled form is wrong when it is not.) It is the non-additivity `ε` alone that makes ordering thermodynamically visible:
`ε > 0` favours alternation (`0101…`), `ε < 0` favours blocks (`000111`), `ε = 0` leaves
residues uncorrelated. **That is the quantity that would give templating a thermodynamic basis
instead of an imposed rule** — see `nonadditivity` and `sequence_correlation_length`.

**Two exact anchors, both checked in `tests/test_thermo.py`.**

1. *Local rule against global state function.* The ΔG of a ligation `a + b -> ab` is the
   **junction bond alone** — every bond interior to `a` or to `b` survives untouched. That
   local rule must agree with the free energy summed over all bonds of `ab`, for **every
   split of every sequence**, and agreeing for every split is Wegscheider consistency: any
   cycle of ligations and cleavages returning to the same composition has `ΔG = 0`.
2. *Flory.* Under an additive assignment the equilibrium length distribution is exactly
   geometric with ratio `ρ = λ_max(T)` and number-average length `1/(1−ρ)` — closed form,
   hit to floating point. `ρ ≥ 1` is runaway polymerisation: **no equilibrium exists**, which
   is a real constraint on parameter choices and not a numerical inconvenience.

**Units and sign.** Free energies are in whatever units `rt` is in; the default `rt = 1.0`
means *energies in units of RT*, which is the safe choice because it removes a whole class of
unit bug. `RT_KJ_PER_MOL_298K` and `RT_KCAL_PER_MOL_298K` are provided for the other
convention. ⚠ **`e` is the free-energy CHANGE on forming a bond, so a stable bond is
NEGATIVE.** Passing a positive "bond strength" gives a polymer that spontaneously falls apart
— plausible numbers rather than an error, which is the failure mode this library is written
against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BondEnergies", "Rates",
    "RT_KJ_PER_MOL_298K", "RT_KCAL_PER_MOL_298K",
    "equilibrium_constant", "rate_constants", "catalysed_rates",
    "enhancement_from_barrier_drop", "barrier_drop_from_enhancement",
    "reaction_free_energies", "reaction_rate_constants",
    "reversible_pairs", "detailed_balance_residual",
    "transfer_matrix", "elongation_ratio", "mean_length",
    "sequence_correlation_length",
]

# R = 8.314462618 J/(mol K) at T = 298.15 K.
RT_KJ_PER_MOL_298K = 2.478967
RT_KCAL_PER_MOL_298K = 0.5924873          # the same quantity / 4.184


@dataclass(frozen=True, eq=False)
class BondEnergies:
    """Free energy of forming one backbone bond, indexed by the two residues it joins.

    `e[i, j]` is the free-energy change on joining `alphabet[i]` on the **left** to
    `alphabet[j]` on the **right**. ⚠ **A stable bond is NEGATIVE**, because this is a change
    and not a strength.

    The matrix is not required to be symmetric. A backbone is directional, and nearest-
    neighbour thermodynamics for real polymers is not orientation-symmetric — the DNA tables
    are 4×4 and unsymmetric. `symmetric()` is the binary convenience constructor the design
    calls for (`E₀₀/E₀₁/E₁₁`, three parameters); the general form is here so that a four-letter
    chemistry needs no redesign.

    `eq=False` deliberately: the default dataclass `__eq__` would compare two arrays and then
    call `bool()` on the elementwise result, which raises rather than answering.
    """

    e: np.ndarray
    alphabet: str = "01"

    def __post_init__(self):
        e = np.asarray(self.e, dtype=float)
        if e.ndim != 2 or e.shape[0] != e.shape[1]:
            raise ValueError(f"e must be a square matrix, got shape {e.shape}")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError(f"alphabet has repeated residues: {self.alphabet!r}")
        if len(self.alphabet) != e.shape[0]:
            raise ValueError(f"alphabet {self.alphabet!r} has {len(self.alphabet)} residues "
                             f"but e is {e.shape[0]}x{e.shape[0]}")
        if not np.all(np.isfinite(e)):
            raise ValueError("e must be finite; an infinite bond energy is a forbidden "
                             "reaction, which belongs in the reaction list, not here")
        object.__setattr__(self, "e", e)

    # -- constructors ----------------------------------------------------------------

    @classmethod
    def symmetric(cls, e00: float, e01: float, e11: float) -> "BondEnergies":
        """The design's three-parameter binary assignment, with `E₀₁ = E₁₀`.

        ⚠ Assuming the 0–1 bond costs the same read either way is a *choice*, and a real
        directional backbone need not obey it. It is the right default here because the
        alternative introduces a fourth parameter that nothing in the model would yet
        distinguish; pass `BondEnergies([[e00, e01], [e10, e11]])` when something does.
        """
        return cls(np.array([[e00, e01], [e01, e11]], dtype=float))

    @classmethod
    def uniform(cls, e: float, alphabet: str = "01") -> "BondEnergies":
        """One bond energy for every pair — Virgo's assignment, and the null case here.

        Its sequence correlation length is zero, so it is the arm to compare a
        non-additive assignment *against*, not a chemistry to draw conclusions from.
        """
        k = len(alphabet)
        return cls(np.full((k, k), float(e)), alphabet=alphabet)

    # -- residues and sequences ------------------------------------------------------

    def _index(self, residue) -> int:
        if isinstance(residue, (int, np.integer)) and not isinstance(residue, bool):
            i = int(residue)
            if not 0 <= i < len(self.alphabet):
                raise ValueError(f"residue index {i} outside alphabet {self.alphabet!r}")
            return i
        pos = self.alphabet.find(str(residue))
        if pos < 0:
            raise ValueError(f"residue {residue!r} is not in alphabet {self.alphabet!r}")
        return pos

    def bond(self, left, right) -> float:
        """Free energy of the single bond joining `left` to `right`."""
        return float(self.e[self._index(left), self._index(right)])

    def sequence(self, s: str) -> float:
        """Free energy of a whole sequence, relative to its separated monomers.

        The sum over its `len(s) - 1` bonds, so a monomer is 0. This is a **state
        function**: it depends on the sequence and not on the order the bonds were made in,
        which is what `ligation` is checked against.

        ⚠ It carries no association term. A per-monomer chemical potential would cancel from
        any ligation (monomer count is conserved) but the standard-state cost of turning two
        molecules into one does not, and that is `dg_assoc`, a parameter of the reaction and
        not of the molecule.
        """
        idx = [self._index(c) for c in s]
        return float(sum(self.e[i, j] for i, j in zip(idx, idx[1:])))

    def ligation(self, a: str, b: str) -> float:
        """Bond free energy of `a + b -> ab` — **the junction bond alone**.

        Every bond interior to `a` or `b` is carried through the reaction untouched, so the
        only term that survives the difference `G(ab) − G(a) − G(b)` is the bond between the
        last residue of `a` and the first of `b`. That is why different *splits* of the same
        molecule have different free energies, and it is where a sequence preference for
        particular cleavage sites comes from.

        Add `dg_assoc` for the full standard reaction free energy; `reaction_free_energies`
        does that for a whole network.
        """
        if not a or not b:
            raise ValueError(f"a ligation needs two non-empty molecules, got {a!r}, {b!r}")
        return self.bond(a[-1], b[0])

    # -- the quantity that matters ---------------------------------------------------

    @property
    def nonadditivity(self) -> float:
        """`ε = E₀₀ + E₁₁ − E₀₁ − E₁₀` — the whole of the sequence preference, in one number.

        When `ε = 0` the bond energy splits as `E[a][b] = h(a) + g(b)`: it says something
        about the left residue and something about the right one and **nothing about the
        pair**, the transfer matrix is singular, and the sequence correlation length is
        exactly zero. So "uniform bond energy admits no sequence preference" is true but not
        tight — the additive family is larger than the uniform one, and every member of it is
        equally blind.

        ⚠ Written `E₀₀ + E₁₁ − 2·E₀₁` while only symmetric matrices were in view, which is
        this expression **only when `E₀₁ = E₁₀`**. Since the matrix is not required to be
        symmetric, that form reported a non-zero preference for genuinely additive chemistries
        like `[[−2, −1], [−4, −3]]`, contradicting `sequence_correlation_length` on the same
        object. The determinant of `exp(−E/RT)` vanishes iff `E₀₀ + E₁₁ = E₀₁ + E₁₀`, and that
        is the condition, symmetric or not.

        `ε > 0` favours alternation, `ε < 0` favours blocks. Binary alphabets only: for a
        larger one the obstruction is a rank condition on the whole matrix, not a scalar.
        """
        if self.e.shape[0] != 2:
            raise ValueError("nonadditivity is defined for a binary alphabet; for "
                            f"{self.alphabet!r} the additive case is rank(exp(-e/rt)) == 1")
        return float(self.e[0, 0] + self.e[1, 1] - self.e[0, 1] - self.e[1, 0])


@dataclass(frozen=True, eq=False)
class Rates:
    """Forward and reverse rate constants of one reaction, or of an array of them.

    Deliberately a pair and never a single number. A forward rate on its own cannot be
    checked against a free energy, and it is exactly that missing half which lets `k_f = 1`
    stand in for a thermodynamics it does not have.
    """

    forward: np.ndarray
    reverse: np.ndarray

    @property
    def equilibrium_constant(self):
        """`k_f / k_r`, which is `exp(-ΔG/RT)` by construction — never fitted separately."""
        return self.forward / self.reverse


def equilibrium_constant(dg, *, rt: float = 1.0):
    """`K = exp(-ΔG/RT)`.

    ⚠ Overflows to `inf` for strongly favourable reactions; the log form is just `-dg/rt`
    and is what to use when the magnitude matters more than the number.
    """
    if rt <= 0:
        raise ValueError(f"rt must be positive, got {rt}")
    return np.exp(-np.asarray(dg, dtype=float) / rt)


def rate_constants(dg, *, barrier, prefactor: float = 1.0, beta: float = 0.5,
                   rt: float = 1.0) -> Rates:
    """Both rate constants of a reaction with standard free energy `dg`.

    The barrier is split between the two directions in the Brønsted / Bell–Evans–Polanyi
    way::

        ΔG‡_forward = barrier + beta·ΔG
        ΔG‡_reverse = barrier − (1−beta)·ΔG
        k = prefactor · exp(−ΔG‡ / RT)

    so that `k_f / k_r = exp(−ΔG/RT)` **identically, for any `barrier` and any `beta`**.
    Detailed balance is a property of the construction here and not a constraint imposed
    afterwards, which is the point: there is no parameter setting that can violate it, and
    therefore no way to accidentally build a network that manufactures free energy.

    `barrier` is the intrinsic barrier — the activation free energy this reaction class would
    have if it were thermoneutral. `beta` is where the transition state sits along the
    reaction coordinate (0 reactant-like, 1 product-like); `0.5` is the symmetric default.
    Both may be arrays over reactions.

    **This is what determines `k_uncat`.** A consistent ΔG cannot coexist with a zero
    uncatalysed rate: fix the barrier and both rates follow, neither of them zero. The
    uncatalysed rate stops being a modelling switch and becomes a consequence.

    ⚠ A barrier lower than `max(−beta·ΔG, (1−beta)·ΔG)` puts the transition state *below* one
    of the two wells, making that direction's rate exceed the prefactor — faster than the
    attempt frequency, which is not a slow reaction with an odd number but an impossible one.
    Raised rather than clipped.
    """
    if rt <= 0:
        raise ValueError(f"rt must be positive, got {rt}")
    # Coerced BEFORE validation, and every operand, not only the ones a scalar path happened
    # to reach: `not 0.0 <= beta <= 1.0` raised "truth value of an array is ambiguous" on the
    # per-reaction arrays this function documents as supported, and validating a coerced copy
    # while computing with the original then failed a step later on a plain list.
    dg = np.asarray(dg, dtype=float)
    barrier = np.asarray(barrier, dtype=float)
    beta = np.asarray(beta, dtype=float)
    prefactor = np.asarray(prefactor, dtype=float)
    if not np.all((beta >= 0.0) & (beta <= 1.0)):
        raise ValueError(f"beta is a position along the reaction coordinate in [0, 1], "
                         f"got {beta}")
    if np.any(prefactor <= 0):
        raise ValueError(f"prefactor must be positive, got {prefactor}")
    needed = np.maximum(-beta * dg, (1.0 - beta) * dg)
    if np.any(barrier < needed):
        worst = float(np.max(needed - barrier))
        raise ValueError(
            f"barrier is below the transition state's own wells by up to {worst:g} — the "
            f"implied rate exceeds the prefactor. With beta={beta} a reaction of free energy "
            f"ΔG needs barrier >= max(-{beta}*ΔG, {1.0 - beta}*ΔG); here that is "
            f"{float(np.max(needed)):g}.")
    return Rates(forward=prefactor * np.exp(-(barrier + beta * dg) / rt),
                 reverse=prefactor * np.exp(-(barrier - (1.0 - beta) * dg) / rt))


def catalysed_rates(rates: Rates, enhancement) -> Rates:
    """Apply a catalytic enhancement to **both** directions, which is the only lawful form.

    A catalyst lowers the barrier. The barrier is shared by the two directions, so both rates
    rise by the same factor and `K_eq` is untouched — the definition of a catalyst, and the
    thing enablement (`k_uncat = 0`) violates: a reaction that cannot happen without its
    catalyst has an equilibrium the catalyst changes, from unreachable to reachable.

    ⚠ **Multiplying only the forward rate is a free-energy source.** Around any cycle it
    returns more work than went in, and the model will simply produce that work, quietly and
    at a plausible-looking rate. `detailed_balance_residual` is the check; this function is
    the shape that makes the check unnecessary.

    `enhancement < 1` is a *slowdown*, and legal — but note it is a different mechanism from
    `rafkit.inhibition`, where an inhibitor blocks a reaction outright rather than scaling it.
    """
    enhancement = np.asarray(enhancement, dtype=float)
    if np.any(enhancement <= 0):
        raise ValueError(f"enhancement must be positive, got {enhancement}; a zero "
                         "enhancement is enablement run backwards and has no barrier")
    return Rates(forward=rates.forward * enhancement,
                 reverse=rates.reverse * enhancement)


def enhancement_from_barrier_drop(drop, *, rt: float = 1.0):
    """Rate enhancement produced by lowering the barrier by `drop`.

    `exp(drop/RT)`. At 298 K a 100× catalyst is 11.4 kJ/mol, roughly one hydrogen bond; the
    10⁶ of a competent enzyme is 34 kJ/mol. Useful for checking that an enhancement chosen
    as a round number corresponds to a barrier a molecule could plausibly supply.
    """
    if rt <= 0:
        raise ValueError(f"rt must be positive, got {rt}")
    return np.exp(np.asarray(drop, dtype=float) / rt)


def barrier_drop_from_enhancement(enhancement, *, rt: float = 1.0):
    """Inverse of `enhancement_from_barrier_drop`: `RT·ln(enhancement)`."""
    if rt <= 0:
        raise ValueError(f"rt must be positive, got {rt}")
    enhancement = np.asarray(enhancement, dtype=float)
    if np.any(enhancement <= 0):
        raise ValueError(f"enhancement must be positive, got {enhancement}")
    return rt * np.log(enhancement)


# ---------------------------------------------------------------------------------------
# Networks. Duck-typed on `.molecules`, `.reactions`, `.directions` like `gillespie` is,
# so a BinaryPolymerNetwork works and nothing is required to inherit anything.
# ---------------------------------------------------------------------------------------

def reaction_free_energies(net, energies: BondEnergies, *, dg_assoc: float = 0.0):
    """Standard free energy of every reaction, **in its stored direction**.

    `net.reactions[r]` is always `(a, b, ab)` whichever way the reaction runs, so the
    junction bond is read off the triple and the sign is taken from `net.directions[r]`.

    `dg_assoc` is the sequence- and length-independent cost of turning two molecules into one
    at the standard state. ⚠ It is the **only** place the standard state enters, and it does
    not cancel the way a per-monomer term does, because a ligation reduces particle count.
    The default `0.0` is a *choice* — "the standard state is such that joining is free" — not
    a neutral absence, and it is what decides whether polymerisation happens at all at unit
    concentration. See `elongation_ratio`.
    """
    mols = net.molecules
    out = np.empty(len(net.reactions), dtype=float)
    for r, (a, b, _ab) in enumerate(net.reactions):
        dg = dg_assoc + energies.ligation(mols[a], mols[b])
        out[r] = dg if net.directions[r] > 0 else -dg
    return out


def reversible_pairs(net) -> tuple[tuple[int, int], ...]:
    """`(ligation index, cleavage index)` for every reaction stored in both directions.

    Matched on the `(a, b, ab)` triple rather than on position, so it does not depend on
    `binary_polymer` happening to lay cleavages out as a second block of the same order.

    Returns empty for a ligation-only chemistry — which is not a clean bill of health but
    the absence of anything to check. An irreversible chemistry has no detailed balance to
    satisfy and, by Gaspard's result, infinite entropy production.
    """
    seen: dict[tuple, dict[int, int]] = {}
    for r, tri in enumerate(net.reactions):
        seen.setdefault(tuple(tri), {})[1 if net.directions[r] > 0 else -1] = r
    return tuple(sorted((v[1], v[-1]) for v in seen.values() if 1 in v and -1 in v))


def _paired(value, name: str, n_reactions: int, pairs) -> np.ndarray:
    """A scalar, or one value per reaction that **agrees across every reversible pair**.

    The two directions of a reversible reaction are one reaction with one transition state.
    Any per-reaction quantity feeding that transition state — the barrier, where along the
    coordinate it sits, the attempt frequency, or a catalyst's factor — must therefore take
    the same value on both halves. Giving them different values does not make one direction
    faster; it makes the reaction a **source of free energy**, silently and at a plausible
    rate. Refused rather than checked afterwards by `detailed_balance_residual`, because by
    then the rates exist and something may already have used them.
    """
    v = np.asarray(value, dtype=float)
    if not v.ndim:
        return v
    if v.shape != (n_reactions,):
        raise ValueError(f"{name} has shape {v.shape}, expected a scalar or one value per "
                         f"reaction ({n_reactions},)")
    bad = [(i, j) for i, j in pairs if not np.isclose(v[i], v[j])]
    if bad:
        i, j = bad[0]
        raise ValueError(
            f"{name} differs across {len(bad)} reversible pair(s), e.g. reactions {i} and "
            f"{j} at {v[i]:g} and {v[j]:g}. The two directions of a reversible reaction share "
            f"one transition state, so a per-reaction {name} must agree on both halves; "
            "differing values make the reaction a source of free energy.")
    return v


def reaction_rate_constants(net, energies: BondEnergies, *, barrier,
                            prefactor: float = 1.0, beta: float = 0.5, rt: float = 1.0,
                            dg_assoc: float = 0.0, enhancement=1.0):
    """Rate constant of every reaction in its stored direction — ready for `propensities`.

    A ligation entry gets the forward branch of the Brønsted split and its cleavage partner
    gets the reverse branch **of the same call**, so the pair satisfies detailed balance
    exactly, whether or not the two entries are adjacent in the array.

    `enhancement` is the factor applied where the reaction is catalysed; pass a scalar, or an
    array over reactions. ⚠ A per-reaction array is **checked against the reversible pairs**
    and refused if the two directions disagree, because differing factors on the two halves
    of one reversible reaction is precisely the free-energy source `catalysed_rates` exists
    to prevent. `binary_polymer(paired_catalysis=True)` already shares catalyst sets across
    the pair, so a mask built from those sets passes.

    ⚠ The same check applies to `barrier`, `beta` and `prefactor`, and for the same reason.
    Checking only `enhancement` left the hole open one door along: a per-reaction *barrier*
    array assigns the two halves of a reversible reaction different transition states, which
    is not a catalyst but is just as much a free-energy source — measured at a residual of
    4.0 before this was closed.

    ⚠ The enhancement here is *static* — the value a reaction takes **when** its catalyst is
    present. Whether it is present is a property of the state, and belongs to the simulator.
    """
    # The ligation-orientation free energy, taken from `reaction_free_energies` and unsigned
    # rather than recomputed, so the junction rule lives in exactly one place.
    directions = np.asarray(net.directions)
    forward = directions > 0
    dg_lig = np.where(forward, 1.0, -1.0) * reaction_free_energies(net, energies,
                                                                  dg_assoc=dg_assoc)
    pairs = reversible_pairs(net)
    n = len(net.reactions)
    barrier = _paired(barrier, "barrier", n, pairs)
    beta = _paired(beta, "beta", n, pairs)
    prefactor = _paired(prefactor, "prefactor", n, pairs)
    enhancement = _paired(enhancement, "enhancement", n, pairs)

    rates = rate_constants(dg_lig, barrier=barrier, prefactor=prefactor, beta=beta, rt=rt)
    k = np.where(forward, rates.forward, rates.reverse)
    # Routed through `catalysed_rates` rather than multiplied here, so that the one
    # place a catalytic factor is applied stays the one place it is validated.
    return catalysed_rates(Rates(forward=k, reverse=k), enhancement).forward


def detailed_balance_residual(net, k, energies: BondEnergies, *, dg_assoc: float = 0.0,
                              rt: float = 1.0) -> float:
    """Largest `|ln(k_f/k_r) + ΔG/RT|` over the network's reversible pairs. Zero is lawful.

    This is the check that catches every way of breaking thermodynamics with rate constants:
    unit rates both ways (`ΔG` is then asserted to be 0, so any non-zero bond energy shows
    up here), a catalyst applied to one direction, or `k_uncat = 0` — which returns `inf`,
    because enablement is an infinite violation and not a large one.

    ⚠ **Raises on a network with no reversible pairs rather than returning 0.** A broken
    query and a clean network would otherwise return the same number, and the number that
    means "nothing to check" must not be the number that means "consistent".
    """
    pairs = reversible_pairs(net)
    if not pairs:
        raise ValueError(
            "no reversible pairs: every reaction is stored in one direction only, so there "
            "is no detailed balance to violate. Returning 0 here would read as 'consistent'. "
            "Generate the chemistry with cleavage=True to make the check meaningful.")
    k = np.asarray(k, dtype=float)
    if k.shape != (len(net.reactions),):
        # Indexed by reaction, so a short `k` raised a bare IndexError and a merely
        # MISALIGNED one returned a residual computed from the wrong reactions.
        raise ValueError(f"k has shape {k.shape}, expected one rate constant per reaction "
                         f"({len(net.reactions)},)")
    dg = reaction_free_energies(net, energies, dg_assoc=dg_assoc)
    worst = 0.0
    for i, j in pairs:
        if k[i] <= 0 or k[j] <= 0:
            return float("inf")           # enablement: not a large violation, an infinite one
        worst = max(worst, abs(float(np.log(k[i] / k[j]) + dg[i] / rt)))
    return worst


# ---------------------------------------------------------------------------------------
# The equilibrium ensemble. A 1-D Ising chain, so it is exactly solvable and the answers
# below are closed form rather than simulated.
# ---------------------------------------------------------------------------------------

def transfer_matrix(energies: BondEnergies, *, rt: float = 1.0, monomer=1.0,
                    dg_assoc: float = 0.0) -> np.ndarray:
    """`T[i, j] = m_j · exp(−(e[i,j] + dg_assoc)/RT)`, the weight of extending by one residue.

    With this normalisation the equilibrium weight of a sequence `s` is
    `m_{s₀} · ∏ T[s_i, s_{i+1}]`, which is the mass-action result for the chain built by any
    sequence of ligations — the same number by every route, which is why it can be a transfer
    matrix at all.

    `monomer` is the equilibrium monomer concentration, scalar or one per residue. Folding it
    into the matrix rather than carrying it alongside is what lets an unequal monomer supply
    be handled with no extra machinery.
    """
    if rt <= 0:
        raise ValueError(f"rt must be positive, got {rt}")
    k = len(energies.alphabet)
    m = np.broadcast_to(np.asarray(monomer, dtype=float), (k,))
    if np.any(m <= 0):
        raise ValueError(f"monomer concentrations must be positive, got {monomer}")
    return m[None, :] * np.exp(-(energies.e + dg_assoc) / rt)


# Relative size below which a subdominant eigenvalue is treated as zero. The additive
# condition e00 + e11 = 2*e01 is an EXACT algebraic cancellation that floating-point `exp`
# does not reproduce: an additive assignment lands at |lambda2|/lambda1 ~ 1e-16 rather than
# at 0, and 1/ln(1e16) = 0.027, so an uncorrelated chemistry reports 0.027 bonds of sequence
# memory. That is a wrong answer wearing plausible numbers, which is the failure this library
# is written against. The floor costs nothing real: the smallest correlation length it can
# suppress decays by e^-28 per bond.
_EIGENVALUE_FLOOR = 1e-12


def _perron_and_subdominant(t: np.ndarray) -> tuple[float, float]:
    """The Perron root of a strictly positive matrix, and the largest **modulus** below it.

    Perron–Frobenius gives a real positive leading eigenvalue, strictly dominant in modulus.
    ⚠ It says nothing about the rest: **a positive matrix of size 3 or more may have complex
    subdominant eigenvalues**, and a randomly drawn 4×4 bond-energy matrix does so on the
    first try. An earlier version rejected those as impossible and refused a perfectly
    ordinary nucleotide chemistry. They are legitimate — a complex conjugate pair means the
    sequence correlation *oscillates* as it decays — and the decay length is set by the
    modulus either way, which is what this returns.

    The 2×2 case is taken in closed form rather than through `eigvals`, because the
    discriminant written as `(t₀₀−t₁₁)² + 4·t₀₁·t₁₀` is manifestly non-negative and gives
    `λ₂ = 0` **exactly** for a uniform matrix, where a general eigensolver returns roundoff.
    Larger alphabets rely on `_EIGENVALUE_FLOOR` instead.
    """
    if t.shape == (2, 2):
        tr = t[0, 0] + t[1, 1]
        disc = np.sqrt((t[0, 0] - t[1, 1]) ** 2 + 4.0 * t[0, 1] * t[1, 0])
        return float((tr + disc) / 2.0), float(abs((tr - disc) / 2.0))
    vals = np.linalg.eigvals(t)
    top = int(np.argmax(np.abs(vals)))
    lead = vals[top]
    if abs(lead.imag) > 1e-9 * abs(lead):
        raise ValueError(f"the dominant eigenvalue {lead} is not real, which Perron–Frobenius "
                         "forbids for a positive matrix; the bond energies are probably not "
                         "finite")
    rest = np.delete(vals, top)
    return float(lead.real), float(np.max(np.abs(rest))) if rest.size else 0.0


def elongation_ratio(energies: BondEnergies, *, monomer=1.0, dg_assoc: float = 0.0,
                     rt: float = 1.0) -> float:
    """`ρ`, the equilibrium factor per added monomer — the leading transfer eigenvalue.

    The equilibrium length distribution falls off as `ρ^L`, so:

    * `ρ < 1` — a convergent distribution, mean length `1/(1−ρ)` (see `mean_length`);
    * `ρ ≥ 1` — **there is no equilibrium**: mass accumulates in the longest species the
      chemistry allows, and any answer computed from such a run is a statement about
      `max_len` rather than about the chemistry.

    So this is the guard on a parameter choice. `dg_assoc` and the monomer supply enter here
    and nowhere else in the ensemble: both scale `T` uniformly, so they move `ρ` but leave
    `sequence_correlation_length` alone.
    """
    return _perron_and_subdominant(transfer_matrix(energies, rt=rt, monomer=monomer,
                                                  dg_assoc=dg_assoc))[0]


def mean_length(energies: BondEnergies, *, monomer=1.0, dg_assoc: float = 0.0,
                rt: float = 1.0) -> float:
    """Number-average polymer length at equilibrium, `1/(1−ρ)` — Flory's result.

    ⚠ **The geometric law starts at the first bond, not at the first molecule.** A monomer has
    no bonds, so its weight is set by the monomer supply alone while every longer species also
    carries bond energies; the step from length 1 to length 2 is therefore a **boundary term**
    and need not equal `ρ`. From length 2 onward the ratio is exactly `ρ` for any additive
    assignment, and asymptotically `ρ` otherwise.

    A uniform assignment hides this — its first step happens to equal `ρ` — which is exactly
    how it would go unnoticed. Measured on the additive-but-not-uniform assignment
    `E = (−2, −2.5, −3)` at `ρ = 0.4`, the first step is 0.377 and this function overstates
    the true number-average by **1.4%**. Exact for a uniform assignment and an even monomer
    supply; the asymptotic value otherwise, and the size of that gap is on the record in
    `tests/test_thermo.py` rather than assumed small.
    """
    rho = elongation_ratio(energies, monomer=monomer, dg_assoc=dg_assoc, rt=rt)
    if rho >= 1.0:
        raise ValueError(
            f"elongation ratio is {rho:g} >= 1: polymerisation runs away and there is no "
            "equilibrium length to average. Lower the monomer concentration, or make "
            "dg_assoc less favourable.")
    return 1.0 / (1.0 - rho)


def sequence_correlation_length(energies: BondEnergies, *, rt: float = 1.0, monomer=1.0,
                                dg_assoc: float = 0.0) -> float:
    """How far along a chain one residue's identity biases the next, in bonds.

    `ξ = 1 / ln(λ₁/|λ₂|)`, the Ising correlation length of the equilibrium sequence
    ensemble. **Exactly zero for every additive assignment** (`nonadditivity == 0`), where
    `λ₂ = 0`: the residues are independent and no ordering is preferred. It is finite and
    positive only when the bond energy says something about the *pair*.

    The sign of `λ₂` — carried by `nonadditivity` — says which kind: `ε > 0` gives `λ₂ < 0`
    and alternating chains, `ε < 0` gives `λ₂ > 0` and blocky ones. This function returns the
    length either way; read `nonadditivity` for the type. ⚠ Beyond a binary alphabet the
    subdominant eigenvalue may be a complex pair, meaning a correlation that oscillates as it
    decays; the length returned is then the decay envelope alone.

    ⚠ Unaffected by `monomer` and `dg_assoc` when the monomer supply is even, since both
    scale the matrix and cancel from the ratio. They are arguments only so that an uneven
    supply, which does not cancel, can be passed.

    ⚠ A subdominant eigenvalue below `_EIGENVALUE_FLOOR` of the leading one is read as zero.
    Without that floor an additive assignment returns 0.027 rather than 0 — roundoff in
    `exp` reported as sequence memory. `nonadditivity` is the exact test, since it is
    computed from the energies themselves and never passes through an exponential.
    """
    lead, sub = _perron_and_subdominant(transfer_matrix(energies, rt=rt, monomer=monomer,
                                                       dg_assoc=dg_assoc))
    if sub <= _EIGENVALUE_FLOOR * abs(lead):
        return 0.0
    return float(1.0 / np.log(abs(lead) / sub))
