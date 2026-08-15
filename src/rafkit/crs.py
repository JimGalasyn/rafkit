"""Read and write CatReNet's CRS format.

CRS is the interchange format of CatReNet (Huson, Xavier & Steel, *Bioinformatics*
40(8) btae515, 2024), the reference implementation of RAF theory. Supporting it means
networks can be round-tripped between this library and the field's existing tooling,
and that any result here can be checked against an independent implementation.

The format is a food line and one line per reaction::

    # a comment
    Food: a, b, aa, ab

    r1 : aa + b [ab,ba] <-> aab
    r2 : ab [aa] => abab

`[...]` lists catalysts, `<->` marks a reversible reaction and `=>` (or `->`) an
irreversible one.

Catalysts are alternatives, any one of which suffices. A braced group is a
**conjunctive** requirement -- `[{a,d}, e]` means *a and d together*, or *e* -- which
is the notation Huson, Xavier & Steel (2024) use. Two edge cases carry meaning and are
not interchangeable: `[]` means the reaction **must** be catalysed and nothing does so,
while `[{}]` means it **may proceed uncatalysed**. A reversible reaction is read as **two** reactions, forward and
reverse, sharing a catalyst set -- which is the reading its own generator uses.

`X + X -> Y` is written `X ... -> Y`, with the repeated reactant collapsed. Since
reactants are sets for every RAF computation, nothing is lost; stoichiometry is not
modelled here either way.

This module talks to CatReNet only through files. No CatReNet code is used or
derived from -- it is GPL v3, and this library is MIT.
"""
from __future__ import annotations

import re
from pathlib import Path

from rafkit.network import ReactionNetwork

_ARROW = re.compile(r"\s*(<->|<=>|=>|->)\s*")
_LINE = re.compile(r"^\s*(?P<name>[^:]+?)\s*:\s*(?P<body>.*)$")


def _split_list(text: str) -> list[str]:
    return [s for s in (t.strip() for t in re.split(r"[,+]", text)) if s]


def _parse_catalysts(text: str) -> list[list[str]]:
    """Parse a catalyst list into alternative sets, honouring braced conjunctions.

    Splitting on commas alone is wrong the moment a braced group appears -- `{a,d}`
    would become `{a` and `d}` -- so groups are pulled out first.
    """
    groups, rest = [], text
    for m in re.finditer(r"\{([^}]*)\}", text):
        groups.append(_split_list(m.group(1)))     # may be empty: {} means "uncatalysed"
    rest = re.sub(r"\{[^}]*\}", " ", text)
    groups += [[name] for name in _split_list(rest)]
    return groups


def parse_crs(text: str) -> ReactionNetwork:
    """Parse CRS text into a `ReactionNetwork`."""
    food_names: list[str] = []
    parsed: list[tuple[str, list[str], list[str], list[str], bool]] = []

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith("food:"):
            food_names += _split_list(line.split(":", 1)[1])
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, body = m.group("name"), m.group("body")

        cats: list[list[str]] = []
        if "[" in body:
            pre, rest = body.split("[", 1)
            inside, post = rest.split("]", 1)
            cats = _parse_catalysts(inside)
            body = pre + " " + post

        arrow = _ARROW.search(body)
        if not arrow:
            continue
        lhs, rhs = body[:arrow.start()], body[arrow.end():]
        parsed.append((name, _split_list(lhs), _split_list(rhs), cats,
                       arrow.group(1) in ("<->", "<=>")))

    # Stable molecule indexing: food first, then order of appearance.
    index: dict[str, int] = {}
    for n in food_names:
        index.setdefault(n, len(index))
    for _, lhs, rhs, cats, _ in parsed:
        for n in (*lhs, *rhs, *(x for g in cats for x in g)):
            index.setdefault(n, len(index))

    pairs, catalysts, names = [], [], []
    for name, lhs, rhs, cats, reversible in parsed:
        cat = frozenset(frozenset(index[c] for c in g) for g in cats)
        fwd = (tuple(index[x] for x in lhs), tuple(index[x] for x in rhs))
        pairs.append(fwd); catalysts.append(cat); names.append(name)
        if reversible:
            pairs.append((fwd[1], fwd[0]))
            catalysts.append(cat)
            names.append(f"{name}_rev")

    molecules = tuple(sorted(index, key=index.get))
    return ReactionNetwork(molecules=molecules,
                           food=frozenset(index[n] for n in food_names),
                           reaction_pairs=tuple(pairs),
                           catalysts=tuple(catalysts),
                           names=tuple(names))


def read_crs(path: str | Path) -> ReactionNetwork:
    """Read a CRS file."""
    return parse_crs(Path(path).read_text())


def to_crs(net, comment: str = "") -> str:
    """Serialise any network exposing the rafkit protocol to CRS text.

    Reversible pairs are **not** re-merged: each stored direction is written as its
    own one-way reaction. That round-trips faithfully and keeps the output honest
    about what the object actually holds.
    """
    out = []
    if comment:
        out += [f"# {line}" for line in comment.splitlines()]
    name = lambda m: net.molecules[m]
    out.append("")
    out.append("Food: " + ", ".join(sorted(name(m) for m in net.food)))
    out.append("")
    names = getattr(net, "names", None) or [f"r{i + 1}" for i in range(net.n_reactions)]
    for r in range(net.n_reactions):
        lhs = " + ".join(dict.fromkeys(name(x) for x in net.reactants(r)))
        rhs = " + ".join(dict.fromkeys(name(x) for x in net.products(r)))
        cats = _format_catalysts(net, r)
        out.append(f"{names[r]} : {lhs} [{cats}] => {rhs}")
    out.append("")
    return "\n".join(out)


def _format_catalysts(net, r: int) -> str:
    """Render a reaction's catalyst sets, using braces only where they are needed."""
    parts = []
    for U in sorted(net.catalysts[r], key=lambda u: sorted(u)):
        names = sorted(net.molecules[c] for c in U)
        parts.append(names[0] if len(names) == 1 else "{" + ",".join(names) + "}")
    return ",".join(parts)


def write_crs(net, path: str | Path, comment: str = "") -> None:
    """Write a network to a CRS file."""
    Path(path).write_text(to_crs(net, comment))
