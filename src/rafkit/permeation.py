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
of radius 0.5 sits in a diffusion voxel of 2.5 × 2.5, a volume ratio of **11.9**. Writing the
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

    `blocked` names species that are held back regardless of size, which is how a treatment
    arm isolates one molecule's transport — the manipulation the induction experiment turns on.
    """
    mask = np.asarray([len(s) if isinstance(s, str) else s for s in lengths]) <= max_len
    mask = np.asarray(mask, dtype=bool).copy()
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

    The result is clipped so an explicit step cannot move more than is present at either end.
    The compartment side binds first whenever it is the smaller volume, since its exchange rate
    `permeability * area / volume_in` is correspondingly higher.
    """
    if volume_in <= 0 or volume_out <= 0:
        raise ValueError(f"volumes must be positive, got in={volume_in}, out={volume_out}")
    n_in = np.asarray(n_in, dtype=float)
    n_out = np.asarray(n_out, dtype=float)
    flux = permeability * area * dt * (n_out / volume_out - n_in / volume_in)
    if permeable is not None:
        flux = flux * np.asarray(permeable, dtype=float)
    return np.clip(flux, -n_in, n_out)
