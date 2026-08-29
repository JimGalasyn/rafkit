"""Export to ANDL, the PetriNuts framework's executable Petri net format.

`rafkit.pnml` makes a network *readable* by the Petri net ecosystem; this module makes it
*runnable*. ANDL (the plain-text format shared by Snoopy, Spike and Marcie) carries what
PNML's ``ptnet`` grammar cannot: stochastic rate constants. A file written here is a
complete stochastic Petri net that Spike executes directly -- an independently developed
simulator re-running this chemistry from its definition, which is the entire point.

**The semantics exported are mass action, and nothing else.** Every transition's
propensity is its rate constant times the falling-factorial count product over its
pre-places -- what Gillespie's direct method computes and what Spike's ``MassAction``
was *measured* to compute (2026-08-29, three-toy calibration; see
``docs/DESIGN_candl_exporter.md`` in the abiogenesis repository):

* a catalyst becomes a **consume-and-produce self-loop**, so it multiplies the
  propensity by its count and is required to be present. This matches chemistries whose
  catalysis is mass action in the catalyst count. It does **not** match
  `rafkit.gillespie`, whose catalysis is a threshold -- any catalyst present buys the
  full rate, and the propensity does not scale with catalyst count. That is a different
  system, and this module makes no attempt to encode it.
* alternative catalyst sets become **separate transitions** (named ``r1``, ``r1#2``, ...)
  whose propensities sum -- which *is* the mass-action reading of "either catalyses":
  each catalytic channel is its own elementary reaction.
* ``a + a -> aa`` is emitted with a weight-2 arc, and Spike computes the standard
  unordered pair count ``n(n-1)/2`` (measured, not assumed -- the alternatives sat
  19-25 SE away). ⚠ This is `rafkit.gillespie`'s convention exactly. It is **not**
  `abiogenesis.stochastic`'s, which counts ordered pairs ``n(n-1)``: a caller mapping
  that chemistry must pre-double identical-reactant rate constants. Two in-house
  conventions differing by 2x on self-pairs -- state which one your rates mean.

Rate constants are emitted as **named constants** (``k_<transition>``), so a Spike
``.spc`` configuration can override any single rate without regenerating the file --
which is what makes a mutation control a config change rather than a code path.

Refused rather than silently altered -- an executable file that drops a feature does not
*document* a different system, it **runs** one:

* **inhibition**: no ANDL representation; `to_andl` raises. (`rafkit.pnml` may annotate
  instead, because a PNML reader sees the annotation; a simulator would not.)
* **a catalyst that is also a reactant of the same reaction**: the self-loop and the
  consuming arc merge into a weight-2 pre-arc, and implementations disagree on the
  combinatorics that implies. Raise, until someone measures what the target tool does.
* ``χ = ∅`` ("must be catalysed, nothing does") with no uncatalysed channel: omitted and
  counted in the header, as in `rafkit.pnml` -- emitting it unconstrained would make an
  impossible reaction fireable.

Stoichiometry is read from the **multiplicity of the reactant/product tuples**.
`ReactionNetwork`'s contract treats those as sets ("it would be for stoichiometry, which
this class does not model"), but both `parse_crs` and `BinaryPolymerNetwork` preserve
duplicates in the tuples they build, so multiplicity is faithful for networks built by
this library. A network whose tuples were deduplicated upstream exports with the wrong
stoichiometry, and nothing here can detect that.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

__all__ = ["to_andl", "write_andl"]


def _identifier(name: str, taken: dict[str, str], prefix: str) -> str:
    """A deterministic ANDL identifier for `name`, unique within `taken`."""
    base = prefix + re.sub(r"\W", "_", name)
    cand, n = base, 1
    while cand in taken and taken[cand] != name:
        n += 1
        cand = f"{base}__{n}"
    taken[cand] = name
    return cand


def _per_reaction(value, n_reactions: int, what: str) -> list[float]:
    if value is None:
        raise ValueError(f"{what} is required: ANDL exists to carry rate constants")
    try:
        seq = [float(value)] * n_reactions
    except TypeError:
        seq = [float(v) for v in value]
        if len(seq) != n_reactions:
            raise ValueError(f"{what} covers {len(seq)} reactions, "
                             f"but the network has {n_reactions}")
    return seq


def _per_molecule(value, net, what: str) -> dict[int, float]:
    """None -> {}; scalar -> every food molecule; mapping name->value -> those."""
    if value is None:
        return {}
    if isinstance(value, dict):
        index = {m: i for i, m in enumerate(net.molecules)}
        missing = [n for n in value if n not in index]
        if missing:
            raise ValueError(f"{what} names unknown molecules: {sorted(missing)}")
        return {index[n]: float(v) for n, v in value.items()}
    return {m: float(value) for m in sorted(net.food)}


def to_andl(net, k, *, name: str = "rafkit", k_uncat=None, marking=None,
            food_influx=None, washout=None) -> str:
    """Serialise a network plus mass-action rate constants to ANDL text.

    `k` (scalar or per-reaction sequence) is the rate constant of each **catalysed
    channel**: one transition per catalyst set, propensity ``k * prod(reactants) *
    prod(catalyst set members)``, summed across alternative sets by construction.

    `k_uncat` (optional, scalar or per-reaction) adds one **uncatalysed transition per
    reaction** at that constant -- the background channel, ``prod(reactants)`` only.
    With it, a ``χ = ∅`` reaction exports as its background channel alone; without it,
    such reactions are omitted and counted in the header.

    `marking` is a ``{molecule name: count}`` dict; molecules not named start at 0,
    except food, which starts at 1 unless overridden (the `rafkit.pnml` default).

    `food_influx` / `washout`: scalar (applied to every food molecule / every molecule
    respectively) or ``{molecule name: rate}``. Influx becomes a source transition with
    an empty preset -- constant propensity equal to the rate. Washout becomes a sink
    transition per molecule, propensity ``rate * count``.
    """
    n_r = net.n_reactions
    ks = _per_reaction(k, n_r, "k")
    kus = _per_reaction(k_uncat, n_r, "k_uncat") if k_uncat is not None else None
    names = getattr(net, "names", None) or [f"r{i + 1}" for i in range(n_r)]

    inhibitors = getattr(net, "inhibitors", ()) or ()
    inhibited = [names[r] for r in range(n_r) if r < len(inhibitors) and inhibitors[r]]
    if inhibited:
        raise ValueError(
            "inhibition has no ANDL representation, and an executable export that "
            "drops it RUNS a different system rather than documenting one; strip the "
            f"inhibitors first if that system is what you want: {inhibited}")

    taken: dict[str, str] = {}
    place = [_identifier(m, taken, "s_") for m in net.molecules]

    marking = dict(marking or {})
    index = {m: i for i, m in enumerate(net.molecules)}
    unknown = [n for n in marking if n not in index]
    if unknown:
        raise ValueError(f"marking names unknown molecules: {sorted(unknown)}")
    counts = {i: 0 for i in range(net.n_molecules)}
    for m in net.food:
        counts[m] = 1
    for n, v in marking.items():
        counts[index[n]] = int(v)

    influx = _per_molecule(food_influx, net, "food_influx")
    outflux = _per_molecule(washout, net, "washout") if isinstance(washout, dict) \
        else ({m: float(washout) for m in range(net.n_molecules)}
              if washout is not None else {})

    constants: list[tuple[str, float]] = []
    transitions: list[tuple[str, str, str]] = []      # (tid, arcs, kname)
    skipped = 0

    def arcs_text(consumed: Counter, produced: Counter) -> str:
        parts = []
        for p in sorted(set(consumed) | set(produced)):
            if consumed.get(p):
                parts.append(f"[{place[p]} - {consumed[p]}]")
            if produced.get(p):
                parts.append(f"[{place[p]} + {produced[p]}]")
        return " & ".join(parts)

    for r in range(n_r):
        reactants = Counter(net.reactants(r))
        products = Counter(net.products(r))
        chi = net.catalysts[r]
        for j, cat_set in enumerate(sorted(chi, key=lambda u: sorted(u))):
            clash = sorted(set(cat_set) & set(reactants))
            if clash:
                raise ValueError(
                    f"reaction {names[r]!r}: catalyst(s) "
                    f"{[net.molecules[c] for c in clash]} are also reactants. The "
                    "self-loop would merge with the consuming arc into a weight-2 "
                    "pre-arc, whose combinatorics differ between implementations; "
                    "refusing until the target tool's convention is measured.")
            tid = _identifier(names[r] if j == 0 else f"{names[r]}#{j + 1}",
                              taken, "t_")
            kname = f"k_{tid}"
            constants.append((kname, ks[r]))
            consumed = reactants + Counter(cat_set)
            produced = products + Counter(cat_set)
            transitions.append((tid, arcs_text(consumed, produced), kname))
        if kus is not None:
            tid = _identifier(f"{names[r]}_u", taken, "t_")
            kname = f"k_{tid}"
            constants.append((kname, kus[r]))
            transitions.append((tid, arcs_text(reactants, products), kname))
        elif not chi:
            skipped += 1

    for m, rate in sorted(influx.items()):
        tid = _identifier(f"src_{net.molecules[m]}", taken, "t_")
        kname = f"k_{tid}"
        constants.append((kname, rate))
        transitions.append((tid, f"[{place[m]} + 1]", kname))
    for m, rate in sorted(outflux.items()):
        tid = _identifier(f"out_{net.molecules[m]}", taken, "t_")
        kname = f"k_{tid}"
        constants.append((kname, rate))
        transitions.append((tid, f"[{place[m]} - 1]", kname))

    notes = [f"Generated by rafkit. {net.n_molecules} species, {n_r} reactions.",
             "Semantics: mass action; catalysts are consume-and-produce self-loops",
             "(propensity scales with catalyst count); identical-reactant pairs use",
             "the unordered convention n(n-1)/2 (Spike, measured 2026-08-29)."]
    if skipped:
        notes.append(f"{skipped} reaction(s) omitted: they require a catalyst, "
                     "nothing catalyses them, and no k_uncat was given, so they "
                     "can never fire.")

    lines = ["/*"] + [f" * {n}" for n in notes] + [" */", "",
             f"spn  [{name}]", "{"]
    if constants:
        lines += ["constants:", "all:"]
        lines += [f"  double {kn} = {kv!r};" for kn, kv in constants]
        lines.append("")
    lines += ["places:", "discrete:"]
    lines += [f"  {place[m]} = {counts[m]};" for m in range(net.n_molecules)]
    lines.append("")
    lines += ["transitions:", "stochastic:"]
    for tid, arcs, kname in transitions:
        lines += [f"  {tid}", "    :", f"    : {arcs}",
                  f"    : MassAction({kname})", "    ;"]
    lines += ["}", ""]
    return "\n".join(lines)


def write_andl(net, k, path: str | Path, **kwargs) -> None:
    """Write a network plus rate constants to an ANDL file."""
    Path(path).write_text(to_andl(net, k, **kwargs))
