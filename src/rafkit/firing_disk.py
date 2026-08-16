"""Serra & Villani's FIRING-DISK construction — a chemistry grown, not enumerated.

Reproduced from *Entropy* 28(2), 184 (2026), §2.2. The contrast with every other generator
here is the point:

* `binary_polymer` and `complementary_polymer` **enumerate** every string up to `max_len`
  and sprinkle catalysis over the result. A species exists because it is short enough.
* the firing disk **grows** outward from a small seed. A species exists only if some
  reaction in the network actually **makes** it.

So a firing-disk chemistry is closed under its own production by construction, where an
enumerated one is full of species nothing can reach. That difference plausibly matters a
great deal for protocell viability, because unreachable species dilute the material without
contributing to it -- which is why this exists.

The loop is theirs: seed a disk of short polymers, designate some as cleavage or
condensation catalysts, find every reaction the current catalysts enable among the current
species, run them, and give each **newly generated** species a chance `p_cat` of being a
catalyst itself -- iterating "until there are no more new reactions or new species to add,
or some termination condition is met."

⚠ **Food is taken to be the firing disk.** The paper does not say what plays the role of
food when such a chemistry is embedded in a protocell, and the disk is the only externally
given set, so that is the reading used here -- an assumption, not a quotation.
"""
from __future__ import annotations

import numpy as np

from rafkit.binary_polymer import BinaryPolymerNetwork, _strings
from rafkit.complementary_polymer import complement


def _site(mol, site_min, site_max, rng):
    width = int(rng.integers(site_min, min(site_max, len(mol)) + 1))
    start = int(rng.integers(0, len(mol) - width + 1))
    site = mol[start:start + width]
    return site, int(rng.integers(1, width))


def firing_disk_polymer(disk_size: int = 24, disk_len: int = 4, max_len: int = 10,
                        p_cat: float = 0.05, p_cleave: float = 0.5,
                        n_cleave_cat: int = 2, n_cond_cat: int = 2,
                        site_min: int = 3, site_max: int = 4,
                        max_species: int = 4000, max_rounds: int = 200,
                        rng: np.random.Generator | None = None) -> BinaryPolymerNetwork:
    """Grow one C-chemistry outward from a firing disk.

    Defaults follow the paper's Figure 3 ensemble where stated: 24 initial species of
    length <= 4, ``Lmax = 10``, ``p_cat = 0.05``, ``p_cleave = 0.5``, active sites uniform
    on 3..4. ``n_cleave_cat`` / ``n_cond_cat`` are their ``NCLini`` / ``NCDini``, whose
    values Figure 3 does not state.

    ``max_species`` and ``max_rounds`` are the "termination condition" their text leaves
    open; both are reported through the returned network's size rather than raising, but a
    run that hits ``max_species`` is a **truncated** chemistry and should be treated as
    such.
    """
    if not 1 <= site_min <= site_max:
        raise ValueError(f"need 1 <= site_min <= site_max, got {site_min}, {site_max}")
    for name, v in (("p_cat", p_cat), ("p_cleave", p_cleave)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} is a probability, got {v}")
    rng = rng or np.random.default_rng()

    pool = [m for m in _strings(disk_len) if len(m) >= 1]
    disk = list(rng.choice(pool, size=min(disk_size, len(pool)), replace=False))
    species: set[str] = set(disk)

    # catalyst -> (site, cut, is_cleaver); seeded per NCLini / NCDini, then grown by p_cat
    cats: dict[str, tuple[str, int, bool]] = {}
    eligible = [m for m in disk if len(m) >= site_min]
    rng.shuffle(eligible)
    for m in eligible[:n_cleave_cat]:
        s, c = _site(m, site_min, site_max, rng); cats[m] = (s, c, True)
    for m in eligible[n_cleave_cat:n_cleave_cat + n_cond_cat]:
        s, c = _site(m, site_min, site_max, rng); cats[m] = (s, c, False)

    # split -> set of catalysts, for each direction; a "split" is the pair (a, b) of ab
    lig: dict[tuple[str, str], set[str]] = {}
    cle: dict[tuple[str, str], set[str]] = {}

    for _ in range(max_rounds):
        fresh: set[str] = set()
        by_pre: dict[str, list[str]] = {}
        by_suf: dict[str, list[str]] = {}
        for m in species:
            for k in range(1, len(m) + 1):
                by_pre.setdefault(m[:k], []).append(m)
                by_suf.setdefault(m[-k:], []).append(m)
        for cat, (site, cut, is_cleaver) in list(cats.items()):
            if is_cleaver:
                target = complement(site)
                for m in list(species):
                    pos = m.find(target)
                    while pos != -1:
                        k = pos + cut
                        if 0 < k < len(m):
                            a, b = m[:k], m[k:]
                            cle.setdefault((a, b), set()).add(cat)
                            fresh.update({a, b} - species)
                        pos = m.find(target, pos + 1)
            else:
                left, right = complement(site[:cut]), complement(site[cut:])
                for a in by_suf.get(left, ()):
                    for b in by_pre.get(right, ()):
                        if len(a) + len(b) <= max_len:
                            lig.setdefault((a, b), set()).add(cat)
                            if a + b not in species:
                                fresh.add(a + b)
        if not fresh:
            break
        for m in sorted(fresh):
            if len(species) >= max_species:
                break
            species.add(m)
            if len(m) >= site_min and rng.random() < p_cat:
                s, c = _site(m, site_min, site_max, rng)
                cats[m] = (s, c, rng.random() < p_cleave)

    # Emit in the paired layout the rest of the library expects: every split that appears
    # in either direction becomes one ligation and one cleavage, each carrying only the
    # catalysts actually assigned to that direction.
    splits = sorted(set(lig) | set(cle))
    splits = [(a, b) for a, b in splits if a in species and b in species
              and a + b in species]
    molecules = tuple(sorted(species, key=lambda m: (len(m), m)))
    index = {m: i for i, m in enumerate(molecules)}
    ligations = tuple((index[a], index[b], index[a + b]) for a, b in splits)
    reactions = ligations + ligations
    directions = (1,) * len(ligations) + (-1,) * len(ligations)
    catalysts = tuple(frozenset(index[c] for c in lig.get(s, ()) if c in index)
                      for s in splits) + \
                tuple(frozenset(index[c] for c in cle.get(s, ()) if c in index)
                      for s in splits)
    food = frozenset(index[m] for m in disk if m in index)
    return BinaryPolymerNetwork(
        molecules=molecules, food=food, reactions=reactions, catalysts=catalysts,
        p=p_cat, max_len=max_len, food_len=disk_len, directions=directions)
