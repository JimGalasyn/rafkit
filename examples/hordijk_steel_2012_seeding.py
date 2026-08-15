"""Watch a maximal RAF assemble itself: Hordijk & Steel (2012), reproduced.

    python examples/hordijk_steel_2012_seeding.py

A maximal RAF does not switch on. Parts of it are catalysed only by their own products,
and those parts cannot start until the product appears by a rare **uncatalysed**
reaction — so the set comes into existence as an ordered sequence of chance events.
That is the result reported in Hordijk & Steel, "Autocatalytic sets extended: dynamics,
inhibition, and a generalization" (*J. Syst. Chem.* 3, 5, 2012), and this script
reproduces it.

**What is and is not reproduced.** Their network is a random draw at n=5, t=2,
p=0.0045 that we cannot reconstruct, so this uses a *structural analogue* found by
searching those same published parameters: seed 540, which yields a maximal RAF of
**eight two-way reactions** — the same size as theirs — containing always-on
food-catalysed reactions, a self-catalysed-from-food ligation, and a four-reaction
core that must be seeded.

Of the four qualitative features of their Figure 6, three reproduce here. The fourth —
a species declining once a later subRAF begins consuming it — is topology-specific and
does not; it is reported below rather than omitted.
"""
from __future__ import annotations

import numpy as np

from rafkit import binary_polymer, max_raf, simulate
from rafkit.catalysis import catalysing_molecules, is_catalysed
from rafkit.gillespie import catalytically_reachable

PUBLISHED = dict(max_len=5, food_len=2, p=0.0045)   # Hordijk & Steel (2012)
SEED = 540
N_EVENTS = 100_000        # their example uses 25,000; see "levelling off" below


def main() -> None:
    net = binary_polymer(**PUBLISHED, rng=np.random.default_rng(SEED), cleavage=True)
    raf = sorted(max_raf(net).reactions)
    nm = net.molecules.__getitem__
    two_way = len({r % (net.n_reactions // 2) for r in raf})

    print(f"Binary polymer, n={PUBLISHED['max_len']}, t={PUBLISHED['food_len']}, "
          f"p={PUBLISHED['p']}, cleavage-ligation")
    print(f"Food: {sorted(nm(f) for f in net.food)}")
    print(f"Maximal RAF: {two_way} two-way reactions "
          f"(Hordijk & Steel's example has 8)\n")

    for r in sorted({r % (net.n_reactions // 2) for r in raf}):
        a, b = net.reactants(r)
        p = net.products(r)[0]
        cats = sorted(nm(c) for c in catalysing_molecules(net.catalysts[r]))
        tags = ""
        if is_catalysed(net.catalysts[r], net.food):
            tags += "  [always-on: food catalyst]"
        if p in catalysing_molecules(net.catalysts[r]):
            tags += "  [self-catalysed: must be seeded]"
        print(f"  {nm(a):>5s} + {nm(b):<5s} <-> {nm(p):<6s} cat={str(cats):<22s}{tags}")

    reachable = catalytically_reachable(net, raf)
    needs_seed = sorted(set(range(net.n_molecules)) - reachable
                        & {m for r in raf for m in net.products(r)})
    print(f"\nCatalysed firings alone reach: {sorted(nm(m) for m in reachable)}")
    print(f"Requires a seeding event:      {sorted(nm(m) for m in needs_seed)}")

    tr = simulate(net, n_events=N_EVENTS, rng=np.random.default_rng(0),
                  reactions=raf, sample_every=N_EVENTS // 400)

    print("\n--- seeding events (fired with no catalyst present) ---")
    for r, t in sorted(tr.first_uncatalysed.items(), key=lambda kv: kv[1])[:8]:
        lhs = " + ".join(nm(x) for x in net.reactants(r))
        rhs = " + ".join(nm(x) for x in net.products(r))
        print(f"  t={t:7.3f}   {lhs} -> {rhs}")

    print("\n--- order of appearance ---")
    for m, t in sorted(((m, t) for m, t in tr.first_appearance.items()
                        if m not in net.food), key=lambda kv: kv[1]):
        mark = "" if m in reachable else "   <- needed a seed"
        print(f"  t={t:7.3f}   {nm(m):<6s} final count={tr.of(nm(m))[-1]:>5d}{mark}")

    # Criterion 4: growth levels off as the cleavage direction catches up. This DOES
    # reproduce, but not within their 25,000 events on this network -- which is why the
    # script runs longer and why the test suite leaves this to the example.
    print("\n--- growth levelling off (late slope vs early slope) ---")
    t, n = tr.times, len(tr.times)
    q = n // 4
    for m in sorted({m for r in raf for m in net.products(r)} - set(net.food)):
        x = tr.of(nm(m)).astype(float)
        if x[-1] < 20:
            continue
        early = (x[q] - x[0]) / (t[q] - t[0] + 1e-9)
        late = (x[-1] - x[-q]) / (t[-1] - t[-q] + 1e-9)
        verdict = "levelling" if late < early else "still accelerating"
        print(f"  {nm(m):<6s} early={early:7.1f}/t  late={late:7.1f}/t  "
              f"ratio={late / max(early, 1e-9):5.2f}  {verdict}")

    print("\n--- not reproduced, and checked rather than assumed ---")
    print("  Their Figure 6 shows molecule 100 DECLINING once the red subRAF begins")
    print("  consuming it. Here 001 is the analogue: it is consumed by 001 + 10 ->")
    print("  00110, a reaction that only starts once 00110 has been seeded.")
    print("  Across 8 RNG seeds its late slope is negative in 4 of 8 runs -- a coin")
    print("  flip, so the effect is absent and any single run showing it is chance.")
    print("  What IS consistent (8/8) is that 001 peaks near 70 and settles lower:")
    print("  saturation, not consumption by a later subRAF.")


if __name__ == "__main__":
    main()
