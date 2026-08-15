"""Inhibition: a network that gains a subRAF, then loses it.

    python examples/inhibition_dissolution.py

Every other example in this library shows autocatalysis *arriving*. This one shows it
leaving, which needs inhibition — a molecule that prevents a reaction rather than
catalysing it. Hordijk, Naylor, Krasnogor & Fellermann (2018) report exactly this in a
spatial protocell world, where inhibitory molecules cause the **loss** of autocatalytic
subsets and thereby make change bidirectional.

Two things are shown, and they are different in kind:

1. **Statically** — the uninhibited RAFs of Hordijk & Steel (2012), which are what an
   inhibited network can sustain *at all*. There is generally more than one and none of
   them is canonical, because inhibition destroys the monotonicity that makes a maximal
   RAF unique.
2. **Dynamically** — a run in which a subRAF produces steadily, a rare uncatalysed
   event brings its inhibitor into existence, and production stops dead.
"""
from __future__ import annotations

import numpy as np

from rafkit import max_raf, parse_crs, simulate
from rafkit.inhibition import classes_from_inhibitors, is_uraf, max_urafs, support

# Each reaction is inhibited by the other's product, so neither can coexist with it.
MUTUAL = """
Food: a, b
r1 : a + b [a] {d} => c
r2 : a + b [b] {c} => d
"""

# r1 is always on. r2 runs as soon as c exists. r3 is catalysed only by its own
# product, so its inhibitor e arrives late, by a rare uncatalysed reaction.
DISSOLVING = """
Food: a, b
r1 : a + b [a]     => c
r2 : a + c [c] {e} => d
r3 : a + b [e]     => e
"""


def statics() -> None:
    net = parse_crs(MUTUAL)
    nm = lambda rs: "{" + ",".join(sorted(net.names[r] for r in rs)) + "}"
    print("A network where each reaction inhibits the other:\n")
    for r in range(net.n_reactions):
        inh = sorted(net.molecules[x] for x in net.inhibitors[r])
        print(f"  {net.names[r]} : inhibited by {inh}")

    print(f"\n  maximal RAF (inhibition ignored, per the definition): {nm(max_raf(net).reactions)}")
    print("  maximal u-RAFs:")
    for u in max_urafs(net):
        print(f"    {nm(u)}   support={sorted(net.molecules[m] for m in support(net, u))}")

    cls = classes_from_inhibitors(net)
    both = frozenset({0, 1})
    print(f"\n  their union {nm(both)} is an RAF, but a u-RAF? {is_uraf(net, both, cls)}")
    print("  -> no unique maximum. Two answers, neither canonical, which is why")
    print("     max_urafs returns a collection rather than a set.")


def dynamics() -> None:
    net = parse_crs(DISSOLVING)
    print("\n\nA subRAF that runs, then stops:\n")
    for r in range(net.n_reactions):
        inh = sorted(net.molecules[x] for x in net.inhibitors[r])
        note = f"   inhibited by {inh}" if inh else ""
        print(f"  {net.names[r]}{note}")

    tr = simulate(net, n_events=4000, rng=np.random.default_rng(2), sample_every=10)
    t_e = tr.first_seen("e")
    d = tr.of("d").astype(float)
    i = int(np.searchsorted(tr.times, t_e))

    print(f"\n  c (r1's product, always-on)      first seen t={tr.first_seen('c'):.3f}")
    print(f"  d (r2's product)                 first seen t={tr.first_seen('d'):.3f}")
    print(f"  e (the inhibitor, needs seeding) first seen t={t_e:.3f}")
    print(f"\n  d at the moment e appeared : {d[i]:.0f}")
    print(f"  d at the end of the run    : {d[-1]:.0f}")
    print(f"  produced after the block   : {d[-1] - d[i]:.0f}")
    print("\n  r2 was not slowed, it was stopped. Inhibition is a block, where an")
    print("  absent catalyst is only a slowdown -- which is why a network can lose")
    print("  a subRAF outright rather than merely running it less often.")


if __name__ == "__main__":
    statics()
    dynamics()
