"""Size-selective permeation across a compartment membrane.

⚠ **Like `dilution`, this is not a RAF algorithm.** It is one line of transport physics, here
because it is a *modelling choice that changes the answer* and is easy to get wrong in a way
that produces plausible numbers rather than errors — which is the criterion the rest of this
library is written to.

RAF work increasingly runs networks inside compartments embedded in a shared medium, and every
such model needs a rule for what crosses the boundary. The rule used here is

    Hordijk, Naylor, Krasnogor & Fellermann, "Population Dynamics of Autocatalytic Sets in a
    Compartmentalized Spatial World", *Life* **8**(3), 33 (2018)

— *"molecules are allowed to permeate compartment membranes if their lengths do not exceed a
certain threshold. Permeation is proportional to the concentration difference."*

**The trap this module exists for.** "Proportional to the concentration difference" is not
"proportional to the count difference", and the two coincide only when compartment and medium
have the same volume. In a spatial model they generally do not: in the paper above a compartment
of radius 0.5 (a sphere, volume 0.524) sits in a diffusion voxel of 2.5 × 2.5 — and since that
world is two-dimensional, with "if the world type is set to 2D then Y is forced to 1", the voxel
is a slab of unit thickness and volume 6.25. **Both are volumes**, and the ratio is **11.9**.
Quote a voxel *area* against a sphere *volume* and the number is dimensionally meaningless. Writing the
flux as `P · (n_out − n_in)` silently asserts the two are the same size.

That is not a cosmetic difference. Reproducing the paper's own induction experiment with every
printed parameter taken from the authors' published input file, the count-difference form
**cannot match both published arms at any permeability**: sweeping it reaches the control value
at an effect ratio of 1.15 on one branch or 2.44 on the other, bracketing the published 1.60
without ever hitting it. The concentration form reproduces both arms (16.5 ± 5.9 against their
16.3, and 27.8 ± 6.0 against their 26.0) with a single free parameter.

**Calibration tier.** This is checked against a published *figure* with one fitted parameter, not
against closed-form conditions. It belongs with the library's reference-implementation and
published-figure checks — not with `dilution`, which is the only analytic anchor here.
"""
from __future__ import annotations

import numpy as np

__all__ = ["permeable_by_length", "permeation_flux"]


def permeable_by_length(lengths, max_len: int, blocked=()) -> np.ndarray:
    """Boolean mask of species small enough to cross a membrane of aperture `max_len`.

    `lengths` is a sequence of species — strings, whose length is taken, or integers already
    giving a length. `blocked` is a sequence of **positional indices** into that sequence,
    naming species held back regardless of size; that is how a treatment arm isolates one
    molecule's transport, the manipulation the induction experiment turns on.
    """
    if isinstance(lengths, str):
        # "011" would otherwise iterate to ('0', '1', '1') and return a per-CHARACTER mask,
        # which is a wrong answer rather than an error.
        raise TypeError("lengths must be a sequence of species, not a single string; "
                        "pass ['011'] rather than '011'")
    mask = np.asarray([len(s) if isinstance(s, str) else s for s in lengths]) <= max_len
    for i in blocked:
        mask[i] = False
    return mask


def permeation_flux(n_in, n_out, *, permeability: float, area: float,
                    volume_in: float, volume_out: float, permeable=None, dt: float = 1.0):
    """Molecules crossing INTO the compartment this step (negative = net efflux).

    Fick's law across a membrane::

        J = permeability * area * (n_out / volume_out  -  n_in / volume_in)

    so `permeability` is a coefficient with units of length/time, not a rate. **The volumes are
    required rather than optional** — passing counts as if they were concentrations is the error
    this module documents, and there is no default that would let it happen silently.

    **Geometry is per-COMPARTMENT.** `permeability`, `area` and the two volumes may each be a
    scalar or an array broadcastable against `n_in.shape[:-1]` -- the compartment axes, with the
    last axis of `n_in` always being species. A whole population of compartments of differing
    size therefore steps at once, which is the case a spatial model needs, where every pore holds
    a cell with its own radius.

    A per-SPECIES vector is refused rather than broadcast, because there is no way to tell one
    from a per-compartment vector when the two counts coincide, and guessing produced a silent
    wrong answer: keying the reshape off `volume_in` alone meant a batched `area` with a scalar
    `volume_in` broadcast along the species axis, giving [1.6, 1.6, 1.6, 1.6] where the right
    answer was [1.6, 3.2, 4.8, 6.4]. Per-species variation belongs in `permeable`, which
    multiplies the flux and so carries graded factors, not only a 0/1 mask.

    The result is clipped so an explicit step cannot move more than is present at either end.
    The compartment side binds first whenever it is the smaller volume, since its exchange rate
    `permeability * area / volume_in` is correspondingly higher.
    """
    volume_in = np.asarray(volume_in, dtype=float)
    volume_out = np.asarray(volume_out, dtype=float)
    permeability = np.asarray(permeability, dtype=float)
    area = np.asarray(area, dtype=float)
    if np.any(volume_in <= 0) or np.any(volume_out <= 0):
        raise ValueError(f"volumes must be positive, got in={volume_in}, out={volume_out}")
    if np.any(permeability < 0) or np.any(area < 0):
        # A negative coefficient silently reverses the flux, which is uphill transport --
        # exactly the plausible-numbers-rather-than-errors failure this module is about.
        raise ValueError(f"permeability and area must be non-negative, "
                         f"got permeability={permeability}, area={area}")
    # atleast_1d: a scalar input would otherwise yield a 0-d array, so `result[0]` raises
    n_in = np.atleast_1d(np.asarray(n_in, dtype=float))
    n_out = np.atleast_1d(np.asarray(n_out, dtype=float))
    batch = n_in.shape[:-1]          # compartment axes; the last axis is always species

    def _per_compartment(x, name):
        """Scalar, or broadcastable against the COMPARTMENT axes, then given a species axis.

        Checking every operand against one stated contract -- rather than reshaping them all
        to follow whichever one happened to be an array -- is what makes a per-species vector
        an error instead of a silently mis-broadcast answer.
        """
        if x.ndim == 0:
            return x
        try:
            x = np.broadcast_to(x, batch)
        except ValueError:
            raise ValueError(
                f"{name} has shape {x.shape}, which is not scalar and does not broadcast "
                f"against the compartment axes {batch} of n_in {n_in.shape}. Geometry is "
                f"per-compartment; for per-species variation use `permeable`, which "
                f"multiplies the flux and accepts graded factors."
            ) from None
        return x[..., None]

    permeability = _per_compartment(permeability, "permeability")
    area = _per_compartment(area, "area")
    volume_in = _per_compartment(volume_in, "volume_in")
    volume_out = _per_compartment(volume_out, "volume_out")
    flux = permeability * area * dt * (n_out / volume_out - n_in / volume_in)
    if permeable is not None:
        flux = flux * np.asarray(permeable, dtype=float)
    return np.clip(flux, -n_in, n_out)
