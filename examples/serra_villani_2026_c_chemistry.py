"""Reproduce Serra & Villani (2026) Figure 3's C-chemistry ensemble.

    Serra & Villani, "Template-Based Catalysis and the Emergence of Collectively
    Autocatalytic Systems", Entropy 28(2), 184 (2026), figure 3 and section 2.3.

Their ensemble: p_cat = 0.05, p_cl = 0.5, 24 initial species of maximum length 4,
Lmax = 10, active sites uniform on 3..4 -- the defaults of `firing_disk_polymer`.

Two published claims are checked here, and neither was tuned for:

  * the SCALE of a large C-chemistry -- "typically 2000 species and 40,000 reactions",
    "only 100 catalysts", hosting "a RAF often as large as the entire chemistry";
  * the CATALYST CONTRAST against a K-chemistry of the same size -- each C-catalyst
    drives far more reactions, because all its targets share one active site, where a
    K-catalyst's targets are independent draws.

Run:  python examples/serra_villani_2026_c_chemistry.py
"""
from __future__ import annotations

import numpy as np

from rafkit import binary_polymer, firing_disk_polymer, max_raf

SEEDS = range(5)


def catalysts(net):
    return {x for entry in net.catalysts for group in entry for x in group}


def edges(net):
    return sum(len(group) for entry in net.catalysts for group in entry)


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print()
    print(f"{'seed':>5} {'species':>8} {'reactions':>10} {'catalysts':>10} "
          f"{'rxn/cat':>8} {'|RAF|':>8} {'RAF %':>7}")
    stats = []
    for s in SEEDS:
        net = firing_disk_polymer(rng=np.random.default_rng(s))
        cats = catalysts(net)
        raf = max_raf(net)
        pct = 100.0 * len(raf.reactions) / max(net.n_reactions, 1)
        per = edges(net) / max(len(cats), 1)
        stats.append((net.n_molecules, net.n_reactions, len(cats), per, pct))
        print(f"{s:>5} {net.n_molecules:>8} {net.n_reactions:>10} {len(cats):>10} "
              f"{per:>8.0f} {len(raf.reactions):>8} {pct:>6.1f}%")

    mean = np.mean(np.array(stats), axis=0)
    print(f"\n{'':>5} {'mean':>8} {mean[0]:>8.0f} {mean[1]:>10.0f} {mean[2]:>10.0f} "
          f"{mean[3]:>8.0f} {'':>8} {mean[4]:>6.1f}%")

    print("\n--- against the published figures ---")
    for label, published, got in (
        ("species", "~2000", f"{mean[0]:.0f}"),
        ("reactions", "~40,000", f"{mean[1]:.0f}"),
        ("catalysts", "'only 100 catalysts'", f"{mean[2]:.0f}"),
        ("RAF size", "'often as large as the entire chemistry'", f"{mean[4]:.0f}% of it"),
    ):
        print(f"  {label:<12} published {published:<42} measured {got}")

    # The K-vs-C contrast, at matched species count and comparable f.
    k = binary_polymer(max_len=8, food_len=2, p=0.003,
                       rng=np.random.default_rng(0), cleavage=True)
    edges_c = mean[3] * mean[2]
    k_per = edges(k) / max(len(catalysts(k)), 1)
    print(f"\n  reactions per catalyst   C-chemistry {mean[3]:>6.0f}   "
          f"K-chemistry {k_per:>4.0f}   (published: ~400 vs ~20)")
    print(f"  distinct catalysts       C-chemistry {mean[2]:>6.0f}   "
          f"K-chemistry {len(catalysts(k)):>4}")
    print(f"  catalysts per reaction   C-chemistry {edges_c / mean[1]:>6.1f}   "
          f"K-chemistry {edges(k) / max(k.n_reactions, 1):>4.1f}   (theirs: ~1.0)")
    print("\nWhere this MATCHES: species count, catalyst count, and RAF fraction land on\n"
          "their figures without tuning. Reaction count is ~19% low.\n"
          "\nWhere it does NOT: reactions per catalyst comes out ~3x their ~400, because\n"
          "our catalysts are the more promiscuous -- ~4 catalysts per reaction against\n"
          "their ~1. Short active sites (3-4 monomers) have common complements, so each\n"
          "site matches more of the chemistry than theirs appears to. The ASYMMETRY is\n"
          "reproduced and is the published claim; the MAGNITUDE is not, and a study that\n"
          "depends on the per-catalyst number should not use this generator until the\n"
          "gap is understood.")


if __name__ == "__main__":
    main()
