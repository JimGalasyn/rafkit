"""E5 — Serra & Villani's C-BPM, where catalysis follows STRUCTURE rather than a coin flip.

Reproduced from Serra & Villani, *Entropy* 28(2), 184 (2026), §2.2, rather than invented:
an in-house structured-catalysis ensemble would be a worse version of a published one.

**The reaction set is identical to the K-BPM's.** Cleavage splits a polymer, condensation
joins two, and the condensation product ``R-M1M2-R'`` is just the concatenation of the two
reactants. What differs is *which catalyst catalyses which reaction*, and that single change
is the whole point of the model:

* a species is a catalyst with probability ``p_cat``, and a catalyst is a **cleavage**
  catalyst with probability ``p_cleave``, otherwise a **condensation** catalyst;
* a catalyst carries an **active site** — a substring of itself, of length ``Lambda`` drawn
  uniformly from ``[site_min, site_max]`` — plus a cut/suture position inside that site;
* it acts on whatever is **complementary** to that site (binary complement, 0<->1).

The consequence, and the reason this ensemble exists: a K-catalyst's targets are
independent draws, while a **C-catalyst's targets all share one template**, so they are
structurally correlated. Serra & Villani measure the signature of that as a much higher and
much more irregular reactions-per-catalyst distribution (~400 against ~20).

Only polymers at least as long as the active site can be catalysts, which falls out of the
construction rather than being imposed.

⚠ **This implements their "total chemistry" mode** — every species up to ``max_len`` exists
and catalysis is assigned over that set. Their alternative **firing-disk** mode grows the
species set outward from a small seed and is a genuinely different construction, not a
parameter of this one; it is not implemented here.
"""
from __future__ import annotations

import numpy as np

from rafkit.binary_polymer import BinaryPolymerNetwork, _strings

_FLIP = str.maketrans("01", "10")


def complement(s: str) -> str:
    """Binary complement — the paper's "an A must correspond to a B, and vice versa"."""
    return s.translate(_FLIP)


def complementary_polymer(max_len: int = 8, food_len: int = 2, p_cat: float = 0.05,
                          p_cleave: float = 0.5, site_min: int = 3, site_max: int = 4,
                          rng: np.random.Generator | None = None) -> BinaryPolymerNetwork:
    """Generate one C-chemistry: catalysis by active-site complementarity.

    Defaults are the paper's Figure 3 ensemble (``p_cat=0.05``, ``p_cleave=0.5``, active
    sites uniform on 3..4). Returns the same `BinaryPolymerNetwork` a K-chemistry does, so
    every downstream consumer -- `max_raf`, the protocell, the u-RAF layer -- is unchanged.
    """
    if max_len < 2:
        raise ValueError(f"max_len must be at least 2, got {max_len}")
    if not 0 <= food_len < max_len:
        raise ValueError(f"food_len must be in [0, max_len), got {food_len}")
    for name, v in (("p_cat", p_cat), ("p_cleave", p_cleave)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} is a probability, got {v}")
    if not 1 <= site_min <= site_max:
        raise ValueError(f"need 1 <= site_min <= site_max, got {site_min}, {site_max}")
    rng = rng or np.random.default_rng()

    molecules = tuple(_strings(max_len))
    index = {m: i for i, m in enumerate(molecules)}
    food = frozenset(i for i, m in enumerate(molecules) if len(m) <= food_len)

    ligations = tuple(
        (index[a], index[b], index[a + b])
        for a in molecules for b in molecules if len(a) + len(b) <= max_len
    )
    reactions = ligations + ligations
    directions = (1,) * len(ligations) + (-1,) * len(ligations)
    n_pairs = len(ligations)
    lig_at = {(a, b): i for i, (a, b, _) in enumerate(ligations)}
    # cleavage of `ab` at offset k is the reverse of the ligation a + b -> ab
    cleave_at = {(index[m], k): n_pairs + lig_at[(index[m[:k]], index[m[k:]])]
                 for m in molecules for k in range(1, len(m))}

    by_prefix: dict[str, list[str]] = {}
    by_suffix: dict[str, list[str]] = {}
    for m in molecules:
        for k in range(1, len(m) + 1):
            by_prefix.setdefault(m[:k], []).append(m)
            by_suffix.setdefault(m[-k:], []).append(m)

    catalysts: list[set[int]] = [set() for _ in reactions]
    for x, mol in enumerate(molecules):
        if len(mol) < site_min or rng.random() >= p_cat:
            continue
        width = int(rng.integers(site_min, min(site_max, len(mol)) + 1))
        start = int(rng.integers(0, len(mol) - width + 1))
        site = mol[start:start + width]
        cut = int(rng.integers(1, width))          # split point inside the active site
        if rng.random() < p_cleave:
            # cleave any polymer carrying a segment complementary to the whole site
            target = complement(site)
            for m in molecules:
                pos = m.find(target)
                while pos != -1:
                    catalysts[cleave_at[(index[m], pos + cut)]].add(x)
                    pos = m.find(target, pos + 1)
        else:
            # join a molecule ENDING complementary to site[:cut] to one STARTING
            # complementary to site[cut:]
            left, right = complement(site[:cut]), complement(site[cut:])
            for a in by_suffix.get(left, ()):
                for b in by_prefix.get(right, ()):
                    if len(a) + len(b) <= max_len:
                        catalysts[lig_at[(index[a], index[b])]].add(x)

    return BinaryPolymerNetwork(
        molecules=molecules, food=food, reactions=reactions,
        catalysts=tuple(frozenset(c) for c in catalysts),
        p=p_cat, max_len=max_len, food_len=food_len, directions=directions)
