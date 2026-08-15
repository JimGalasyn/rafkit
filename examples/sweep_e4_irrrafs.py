"""Count the irreducible RAFs in E4 -- how many lineages the chemistry can carry.

An irreducible RAF is a self-sustaining core with no self-sustaining proper subset:
the minimal thing a propagule must carry to re-establish a network from food alone.
So the number of *distinct* irreducible cores inside one fixed chemistry upper-bounds
the number of distinguishable lineages, and therefore bounds the heritable
information available to any ecology built on top -- the Vasas, Szathmary & Santos
(PNAS 2010) evolvability objection, made computable.

**If the count is 1, there is nothing to inherit** and no pore-colonisation ecology is
possible regardless of what dynamics are added later. That is the kill this script
exists to attempt.

Two readings are reported side by side and must not be conflated:

* `raf`    -- the literal Hordijk & Steel condition. Includes **food-catalysed** cores,
              which run wherever the food runs, need none of their own products, and
              carry no heredity at all (`raf.is_food_catalysed`).
* `strict` -- catalysts must be non-food products. The self-referential reading, and
              the only one in which a core is something a propagule has to deliver.

`strict` is the PRIMARY. Sampling is by randomised shrinking, so a distinct count is
always a LOWER bound; `saturation` records the running distinct-count so a genuine
plateau can be told apart from undersampling.

    .venv/bin/python examples/sweep_e4_irrrafs.py --seeds 8 --samples 20
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from morphospace.chemistry import binary_polymer, max_raf
from morphospace.chemistry.raf import (irrraf_census, is_food_catalysed,
                                       max_raf_strict, sample_irrraf)


def _census(net, base, n_samples, rng, strict):
    """Census plus the running distinct-count, so undersampling is visible."""
    if base.is_empty:
        return {"n_distinct": 0, "n_food_catalysed": 0, "sizes": [], "saturation": [],
                "mean_size": float("nan"), "mean_jaccard": float("nan"),
                "core_size": 0, "union_size": 0}
    seen, saturation, found = set(), [], []
    for _ in range(n_samples):
        c = sample_irrraf(net, base.reactions, rng, strict=strict)
        found.append(c)
        seen.add(c)
        saturation.append(len(seen))
    distinct = sorted(seen, key=len)
    jac = [len(a & b) / len(a | b)
           for i, a in enumerate(distinct) for b in distinct[i + 1:]]
    core = distinct[0]
    for c in distinct[1:]:
        core = core & c
    return {"n_distinct": len(distinct),
            "n_food_catalysed": sum(1 for c in distinct if is_food_catalysed(net, c)),
            "sizes": [len(c) for c in distinct],
            "saturation": saturation,
            "mean_size": float(np.mean([len(c) for c in found])),
            "mean_jaccard": float(np.mean(jac)) if jac else float("nan"),
            "core_size": len(core),
            "union_size": len(frozenset().union(*distinct))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--food-len", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--n-p", type=int, default=10)
    ap.add_argument("--p-lo", type=float, default=1e-3)
    ap.add_argument("--p-hi", type=float, default=1e-2)
    ap.add_argument("--out", default="campaign_out/e4_irrrafs.json")
    args = ap.parse_args()

    ps = np.logspace(np.log10(args.p_lo), np.log10(args.p_hi), args.n_p)
    rows = []
    print(f"{'cat/mol':>8s} {'|RAF|':>7s} {'|sRAF|':>7s} "
          f"{'distinct(strict)':>17s} {'foodcat':>8s} {'core':>6s} {'jac':>6s} "
          f"{'distinct(raf)':>14s} {'sat?':>5s}")
    for p in ps:
        per_seed = []
        t0 = time.time()
        for s in range(args.seeds):
            net = binary_polymer(max_len=args.max_len, food_len=args.food_len, p=p,
                                 rng=np.random.default_rng(s))
            rng = np.random.default_rng(10_000 + s)
            strict_c = _census(net, max_raf_strict(net), args.samples, rng, True)
            loose_c = _census(net, max_raf(net), args.samples, rng, False)
            per_seed.append({"seed": s,
                             "cat_per_mol": net.mean_catalysed_per_molecule,
                             "raf_size": max_raf(net).size,
                             "strict_raf_size": max_raf_strict(net).size,
                             "strict": strict_c, "loose": loose_c})
        # nanmean: a seed with a single distinct core has no pairwise Jaccard, and
        # plain mean would let that one nan swallow the whole row.
        gm = lambda f: float(np.nanmean([f(r) for r in per_seed]))
        # saturated when the last quarter of samples added no new core, every seed
        sat = all(r["strict"]["saturation"][-1] == r["strict"]["saturation"][
                      max(0, int(0.75 * args.samples) - 1)]
                  for r in per_seed if r["strict"]["saturation"])
        row = {"p": float(p),
               "cat_per_mol": gm(lambda r: r["cat_per_mol"]),
               "raf_size": gm(lambda r: r["raf_size"]),
               "strict_raf_size": gm(lambda r: r["strict_raf_size"]),
               "strict_distinct": gm(lambda r: r["strict"]["n_distinct"]),
               "strict_core_size": gm(lambda r: r["strict"]["core_size"]),
               "strict_mean_size": gm(lambda r: r["strict"]["mean_size"]),
               "strict_jaccard": gm(lambda r: r["strict"]["mean_jaccard"]),
               "loose_distinct": gm(lambda r: r["loose"]["n_distinct"]),
               "loose_food_catalysed": gm(lambda r: r["loose"]["n_food_catalysed"]),
               "saturated": bool(sat),
               "seconds": time.time() - t0,
               "per_seed": per_seed}
        rows.append(row)
        print(f"{row['cat_per_mol']:8.2f} {row['raf_size']:7.0f} "
              f"{row['strict_raf_size']:7.0f} {row['strict_distinct']:17.1f} "
              f"{row['loose_food_catalysed']:8.1f} {row['strict_core_size']:6.1f} "
              f"{row['strict_jaccard']:6.3f} {row['loose_distinct']:14.1f} "
              f"{str(row['saturated']):>5s}", flush=True)

    net0 = binary_polymer(max_len=args.max_len, food_len=args.food_len, p=0.0)
    meta = {"max_len": args.max_len, "food_len": args.food_len, "seeds": args.seeds,
            "samples": args.samples, "n_molecules": net0.n_molecules,
            "n_reactions": net0.n_reactions}
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"\n{meta}\nwrote {out}")


if __name__ == "__main__":
    main()
