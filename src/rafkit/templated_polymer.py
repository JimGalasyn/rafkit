"""Templated ligation: catalysis by COMPLEMENTARITY TO THE PRODUCT, with no random draw.

`binary_polymer` assigns catalysts by a coin flip and `complementary_polymer` by a random
active site per catalyst. Here nothing is drawn: a species templates the ligation
``a + b -> ab`` exactly when it carries the complement of what the ligation makes, so the
catalysis graph is a FUNCTION of the sequence set. That is the content of a template
world, and the reason this ensemble has no ``rng`` argument.

**The rule.** Under an orientation ``f`` -- ``antiparallel``, the reverse complement, or
``parallel``, the plain complement -- a species ``t`` templates ``a + b -> ab`` iff both
reactants reach the per-side overlap, ``len(a) >= h`` and ``len(b) >= h``, and ``t``
contains

* ``rule="product"``: ``f(ab)``, the complement of the whole product;
* ``rule="junction"``: ``f(a[-h:] + b[:h])``, the ``2h`` residues spanning the junction.

The cleavage ``ab -> a + b`` carries its ligation's templates (PAIRED), so a template
changes a reversible reaction's rate and never its equilibrium.

**Why two rules, and which to use.** On a COMPLETE sequence set the junction rule
saturates: at ``max_len`` 7 every eligible reaction is templated by 194 species at
``h`` 1 and 46 at ``h`` 2 (catalysis level f = 978 and 142, against ~5 for a random
chemistry at the RAF threshold), which is a uniform speed-up with no sequence
specificity left. The product rule's template count FALLS with product length -- it is
the number of superstrings up to ``max_len``, 46 for a 4-mer down to exactly one for a
7-mer, its own (reverse) complement -- and f is 12.9 at ``h`` 2. ``"product"`` is the
default for that reason. ⚠ That profile is set by ``max_len - len(ab)`` (to the few
self-overlapping targets that have fewer superstrings): its endpoint is the truncation's,
not chemistry's.

**What the RAF framing does and does not say here.** A templated reaction needs
reactants of length ``>= h``, and only UNTEMPLATED reactions make those from a food set
shorter than ``h``. So at ``h`` 3 with ``food_len`` 2 the maximal RAF is empty, under
either rule, although the chemistry is perfectly runnable on an uncatalysed background.
`max_raf` remains the right question about closure and the wrong one about whether
templating acts.

**The nulls** take a network and return one with the same reactions and a different
catalysis graph, each holding something different fixed -- see their docstrings, and
`catalysis_motifs` for the numbers they are compared on.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from rafkit.binary_polymer import BinaryPolymerNetwork, _strings, binary_polymer
from rafkit.catalysis import catalysing_molecules
from rafkit.complementary_polymer import complement, complementary_polymer
from rafkit.raf import max_raf

ORIENTATIONS = ("antiparallel", "parallel")
RULES = ("product", "junction")


def _target(left: str, right: str, h: int, orientation: str, rule: str) -> str | None:
    """What a template of ``left + right -> left right`` must contain; None if ineligible."""
    if h < 1:
        raise ValueError(f"h is a per-side overlap of at least 1, got {h}")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {ORIENTATIONS}, got {orientation!r}")
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    if len(left) < h or len(right) < h:
        return None
    window = left + right if rule == "product" else left[-h:] + right[:h]
    paired = complement(window)
    return paired[::-1] if orientation == "antiparallel" else paired


def templated_catalysts(left: str, right: str, species: Sequence[str], *, h: int = 2,
                        orientation: str = "antiparallel",
                        rule: str = "product") -> frozenset[int]:
    """Indices into `species` of the strands that template ``left + right -> left right``.

    The rule itself, as a pure function over whatever strands EXIST -- so a simulator
    holding explicit strands and `templated_polymer` holding the complete set apply the
    same rule by construction rather than by re-implementation.
    """
    target = _target(left, right, h, orientation, rule)
    if target is None:
        return frozenset()
    return frozenset(i for i, t in enumerate(species) if target in t)


def templated_polymer(max_len: int = 7, food_len: int = 2, *, h: int = 2,
                      orientation: str = "antiparallel", rule: str = "product",
                      template_class: Callable[[str], bool] | None = None
                      ) -> BinaryPolymerNetwork:
    """The complete binary-polymer chemistry with templated catalysis. Deterministic.

    Same molecules, reactions and layout as ``binary_polymer(cleavage=True)`` -- every
    ligation, then every cleavage -- so `max_raf` and every other consumer is unchanged;
    only the catalysts differ. ``template_class`` restricts which species may template
    (None: all); a class with no member leaves the network uncatalysed.

    ``p`` is recorded as 0.0: there is no catalysis probability. Use `catalysis_motifs`
    for the level f.
    """
    if max_len < 2:
        raise ValueError(f"max_len must be at least 2, got {max_len}")
    if not 0 <= food_len < max_len:
        raise ValueError(f"food_len must be in [0, max_len), got {food_len}")
    _target("0" * h if h >= 1 else "0", "0", h, orientation, rule)        # validate once

    molecules = tuple(_strings(max_len))
    index = {m: i for i, m in enumerate(molecules)}
    food = frozenset(i for i, m in enumerate(molecules) if len(m) <= food_len)
    may = [True] * len(molecules) if template_class is None else [bool(template_class(m)) for m in molecules]

    # every species by the substrings it contains: one lookup per reaction rather than a
    # scan of the species list (2,046 species x 18,434 ligations at max_len 10)
    containing: dict[str, set[int]] = {}
    for i, m in enumerate(molecules):
        if not may[i]:
            continue
        for start in range(len(m)):
            for stop in range(start + 1, len(m) + 1):
                containing.setdefault(m[start:stop], set()).add(i)

    pairs = [(a, b) for a in molecules for b in molecules if len(a) + len(b) <= max_len]
    ligations = tuple((index[a], index[b], index[a + b]) for a, b in pairs)
    drawn = []
    for a, b in pairs:
        target = _target(a, b, h, orientation, rule)
        drawn.append(frozenset() if target is None else frozenset(containing.get(target, ())))
    return BinaryPolymerNetwork(
        molecules=molecules, food=food, reactions=ligations + ligations,
        catalysts=tuple(drawn) + tuple(drawn), p=0.0, max_len=max_len, food_len=food_len,
        directions=(1,) * len(ligations) + (-1,) * len(ligations))


# --- the catalysis graph, on the reversible PAIR ------------------------------------------------

def _n_pairs(net: BinaryPolymerNetwork) -> int:
    n = net.n_reactions - net.n_cleavages
    if net.n_cleavages not in (0, n):
        raise ValueError("expected every ligation, optionally followed by every cleavage: "
                         f"{n} ligations and {net.n_cleavages} cleavages")
    return n


def _pair_catalysts(net: BinaryPolymerNetwork) -> list[frozenset[int]]:
    """Per reversible pair, the molecules catalysing either direction."""
    n = _n_pairs(net)
    out = []
    for i in range(n):
        both = catalysing_molecules(net.catalysts[i])
        if net.n_cleavages:
            both = both | catalysing_molecules(net.catalysts[i + n])
        out.append(both)
    return out


def _simple_paired_edges(net: BinaryPolymerNetwork) -> list[tuple[int, int]]:
    """The (template, pair) edges of a network whose catalysis is disjunctive and paired --
    what a rewiring null can preserve. Anything else is refused rather than flattened."""
    n = _n_pairs(net)
    for i in range(n):
        if any(len(group) != 1 for group in net.catalysts[i]):
            raise ValueError("a rewiring null needs disjunctive (singleton) catalyst sets")
        if net.n_cleavages and net.catalysts[i] != net.catalysts[i + n]:
            raise ValueError("a rewiring null needs paired catalysis: a ligation and its "
                             "cleavage sharing one catalyst set")
    return [(t, i) for i, c in enumerate(_pair_catalysts(net)) for t in sorted(c)]


def _with_pair_catalysts(net: BinaryPolymerNetwork, cats: Sequence[frozenset[int]]
                         ) -> BinaryPolymerNetwork:
    drawn = tuple(frozenset(c) for c in cats)
    return BinaryPolymerNetwork(
        molecules=net.molecules, food=net.food, reactions=net.reactions,
        catalysts=drawn + drawn if net.n_cleavages else drawn, p=net.p, max_len=net.max_len,
        food_len=net.food_len, directions=net.directions, inhibitors=net.inhibitors)


def catalysis_motifs(net: BinaryPolymerNetwork) -> dict:
    """What a templated chemistry and its nulls are compared on, counted on the reversible
    pair (a ligation and its cleavage are ONE reaction, as in `catalysis_level`).

    ==========================  ==========================================================
    ``edges``                   (molecule, pair) catalysis edges
    ``f``                       edges per molecule -- the percolation variable
    ``reach``                   fraction of pairs with at least one catalyst
    ``per_reaction``            edges per CATALYSED pair
    ``by_product_length``       mean catalysts per catalysed pair, by product length
    ``self_reactions``          pairs catalysed by their own ligation product
    ``self_products``           distinct products among those -- the direct autocatalysts
    ``pairs``                   unordered pairs of distinct molecules, each catalysing
                                some reaction that MAKES the other
    ``raf_reactions``           size of the maximal RAF (both directions counted)
    ``self_reactions_by_length``, ``self_products_by_length``
                                the two self counts by the product's length
    ``pairs_by_lengths``        the mutual pairs by their members' lengths, ``(short, long)``
    ==========================  ==========================================================

    ⚠ ``self_reactions`` counts reactions, not products: a 6-mer has five junctions.
    ⚠ Under the product rule a mutual pair is necessarily a sequence and its (reverse)
    complement, so on a complete sequence set ``pairs`` is fixed by the set -- (molecules
    of length >= 2h, minus the self-complementary) / 2 -- and is a CONTRAST for the nulls,
    not a finding about the rule.
    """
    cats = _pair_catalysts(net)
    n = len(cats)
    product = [net.reactions[i][2] for i in range(n)]
    edges = sum(len(c) for c in cats)
    catalysed = [i for i in range(n) if cats[i]]
    by_len: dict[int, list[int]] = {}
    for i in catalysed:
        by_len.setdefault(len(net.molecules[product[i]]), []).append(len(cats[i]))
    selfs = [i for i in catalysed if product[i] in cats[i]]
    makes: dict[int, set[int]] = {}                    # molecule -> the products it helps make
    for i in catalysed:
        for t in cats[i]:
            makes.setdefault(t, set()).add(product[i])
    mutual = [(t, q) for t, made in makes.items() for q in made if t < q and t in makes.get(q, ())]
    pairs = len(mutual)
    length = [len(m) for m in net.molecules]

    def tally(keys):
        out: dict = {}
        for k in keys:
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))
    return dict(
        edges=edges, f=edges / net.n_molecules if net.n_molecules else 0.0,
        reach=len(catalysed) / n if n else 0.0,
        per_reaction=edges / len(catalysed) if catalysed else 0.0,
        by_product_length={L: float(np.mean(v)) for L, v in sorted(by_len.items())},
        self_reactions=len(selfs), self_products=len({product[i] for i in selfs}),
        pairs=pairs, raf_reactions=max_raf(net).size,
        self_reactions_by_length=tally(length[product[i]] for i in selfs),
        self_products_by_length=tally(length[q] for q in {product[i] for i in selfs}),
        pairs_by_lengths=tally(tuple(sorted((length[t], length[q]))) for t, q in mutual))


# --- the nulls ----------------------------------------------------------------------------------

def degree_preserving_null(net: BinaryPolymerNetwork, rng: np.random.Generator, *,
                           stratified: bool = False, swaps_per_edge: int = 20
                           ) -> BinaryPolymerNetwork:
    """Rewire WHICH molecule templates which reaction, keeping every degree.

    A double-edge-swap Markov chain on the bipartite (molecule, pair) graph: two edges
    ``(t1, r1), (t2, r2)`` become ``(t1, r2), (t2, r1)`` unless that would duplicate an
    edge. Both degree sequences are preserved exactly and, unlike stub matching with
    rejection of multi-edges, without bias at high degree. **Holds fixed:** f, the reach,
    and each reaction's catalyst COUNT -- so the profile by product length survives.

    ``stratified=True`` swaps only between edges whose templates have EQUAL LENGTH, so
    each reaction also keeps the LENGTHS of its templates. Under the product rule a
    template is never shorter than the product it templates; the plain shuffle breaks
    that tie along with the sequence tie, and a long product then meets short -- and far
    more abundant -- templates. Measured in a stochastic pore of ~130 molecules: the
    plain shuffle templates a 7-mer product twice as often as the rule does, the
    stratified one as often. Use the stratified null to ask about SEQUENCE; keep the
    plain one beside it to see what length alone does.

    ``swaps_per_edge`` is the number of PROPOSALS per edge; a proposal that would
    duplicate an edge, or cross strata, is skipped.
    """
    edges = _simple_paired_edges(net)
    have, n = set(edges), len(edges)
    length = [len(m) for m in net.molecules]
    for _ in range(swaps_per_edge * n if n > 1 else 0):
        i, j = (int(x) for x in rng.integers(n, size=2))
        (t1, r1), (t2, r2) = edges[i], edges[j]
        if t1 == t2 or r1 == r2 or (t1, r2) in have or (t2, r1) in have:
            continue
        if stratified and length[t1] != length[t2]:
            continue
        have -= {edges[i], edges[j]}
        edges[i], edges[j] = (t1, r2), (t2, r1)
        have |= {edges[i], edges[j]}
    cats: list[set[int]] = [set() for _ in range(_n_pairs(net))]
    for t, r in edges:
        cats[r].add(t)
    return _with_pair_catalysts(net, [frozenset(c) for c in cats])


def motif_matched_null(net: BinaryPolymerNetwork, rng: np.random.Generator, *,
                       match_lengths: bool = False) -> BinaryPolymerNetwork:
    """Random catalysis with `net`'s edge count and EXACTLY its self-catalysed reactions
    and mutual pairs (`catalysis_motifs`), and nothing else of its structure.

    ⚠ Two counts are a thin description of a motif. Planted anywhere, the product rule's
    28 self-catalysed reactions land on ~27 products of every length where the rule has
    12 (four 4-mers, eight 6-mers), and its 114 pairs -- always two strands of EQUAL
    length -- on every mix of lengths, dimers included; a pair with an always-present
    dimer in it is not the dynamical object a 7-mer and its complement are. With the
    default this is a second random-at-f chemistry with two counts pinned.
    ``match_lengths=True`` plants the motifs BY LENGTH: the same number of
    self-catalysing products and reactions at each product length, and the same number
    of pairs at each pair of member lengths (``self_products_by_length``,
    ``self_reactions_by_length``, ``pairs_by_lengths``). Still one junction a planted
    edge, where the rule templates every eligible junction of a partner; and the
    remainder still spreads over every reaction and every template length.

    The motifs are planted first -- a product on that many of its own reactions; that
    many pairs of molecules, each on one reaction making the other -- and the remaining
    edges are drawn uniformly, an edge that would create a further self-catalysed
    reaction or a further mutual pair being redrawn. **Holds fixed:** f and the two motif
    counts. Not the reach, not any degree. (A uniformly random graph at a templated f is
    not motif-free: at f ~ 13 on 254 species it carries some 80 mutual pairs by chance,
    which is why the remainder is constrained rather than left to add its own.)
    """
    target = catalysis_motifs(net)
    n = _n_pairs(net)
    _simple_paired_edges(net)                               # refuse what cannot be matched
    product = [net.reactions[i][2] for i in range(n)]
    making: dict[int, list[int]] = {}
    for i, q in enumerate(product):
        making.setdefault(q, []).append(i)
    makeable = sorted(making)
    cats: list[set[int]] = [set() for _ in range(n)]
    makes: dict[int, set[int]] = {}

    def add(t: int, r: int) -> None:
        cats[r].add(t)
        makes.setdefault(t, set()).add(product[r])

    length = [len(m) for m in net.molecules]
    by_len: dict[int, list[int]] = {}
    for q in makeable:
        by_len.setdefault(length[q], []).append(q)
    if match_lengths:
        for L, n_products in target["self_products_by_length"].items():
            chosen = [by_len[L][int(i)] for i in rng.choice(len(by_len[L]), size=n_products, replace=False)]
            first = [int(rng.choice(making[q])) for q in chosen]          # every product at least once
            rest = [r for q in chosen for r in making[q] if r not in first]
            extra = target["self_reactions_by_length"][L] - n_products
            if extra > len(rest):
                raise ValueError(f"{n_products} products of length {L} have too few reactions for "
                                 f"{target['self_reactions_by_length'][L]} self-catalysed ones")
            for r in first + [rest[int(i)] for i in rng.choice(len(rest), size=extra, replace=False)]:
                add(product[r], r)
        wanted = [lens for lens, k in target["pairs_by_lengths"].items() for _ in range(k)]
    else:
        for r in rng.choice(n, size=target["self_reactions"], replace=False):
            add(product[int(r)], int(r))
        wanted = [None] * target["pairs"]
    for lens in wanted:
        while True:
            if lens is None:
                a, b = (makeable[int(x)] for x in rng.choice(len(makeable), size=2, replace=False))
            else:
                a = by_len[lens[0]][int(rng.integers(len(by_len[lens[0]])))]
                b = by_len[lens[1]][int(rng.integers(len(by_len[lens[1]])))]
            if a == b or b in makes.get(a, ()) or a in makes.get(b, ()):
                continue
            add(a, int(rng.choice(making[b])))
            add(b, int(rng.choice(making[a])))
            break
    left = target["edges"] - sum(len(c) for c in cats)
    if left < 0:
        raise ValueError("the motifs alone need more edges than the network has")
    while left:
        t, r = int(rng.integers(net.n_molecules)), int(rng.integers(n))
        q = product[r]
        if t in cats[r] or t == q:
            continue
        if q not in makes.get(t, ()) and t in makes.get(q, ()):        # would close a new pair
            continue
        add(t, r)
        left -= 1
    return _with_pair_catalysts(net, [frozenset(c) for c in cats])


def _is_complete_bpm(net: BinaryPolymerNetwork) -> None:
    if net.molecules != tuple(_strings(net.max_len)) or not net.n_cleavages:
        raise ValueError("a matched-f ensemble is generated over the complete binary-polymer "
                         "set with cleavage; this network is not that")


def matched_f_random(net: BinaryPolymerNetwork, rng: np.random.Generator
                     ) -> BinaryPolymerNetwork:
    """Kauffman's random chemistry at `net`'s catalysis level, IN EXPECTATION.

    ``binary_polymer(cleavage=True)`` with ``p = edges / (molecules x pairs)``. **Holds
    fixed:** f only, and that on average -- the edge count is binomial. ⚠ f is the right
    percolation variable and a poor description of a matched arm: at a templated f the
    random chemistry spreads its edges over nearly every reaction (reach ~0.92 against a
    product-rule chemistry's 0.61 at ``max_len`` 7).
    """
    _is_complete_bpm(net)
    edges = catalysis_motifs(net)["edges"]
    p = edges / (net.n_molecules * _n_pairs(net))
    return binary_polymer(max_len=net.max_len, food_len=net.food_len, p=p, rng=rng,
                          cleavage=True, paired_catalysis=True)


def matched_f_cbpm(net: BinaryPolymerNetwork, rng: np.random.Generator, *,
                   pilots: int = 8, **cbpm) -> BinaryPolymerNetwork:
    """Serra & Villani's C-BPM at `net`'s catalysis level, IN EXPECTATION.

    C-BPM's expected edge count is linear in ``p_cat`` (each eligible species is a
    catalyst independently), so ``p_cat`` is set from the mean edge count of ``pilots``
    chemistries at ``p_cat = 1``, drawn from `rng` before the one returned. Further
    keywords go to `complementary_polymer` (``p_cleave``, ``site_min``, ``site_max``).
    ⚠ A C-catalyst acts on ONE direction of a reaction: this null's catalysis is
    unpaired where a templated chemistry's is paired, and the edge count matched is the
    pair's (either direction), as in `catalysis_level`.
    """
    _is_complete_bpm(net)
    if "p_cat" in cbpm:
        raise ValueError("p_cat is what this function sets")
    edges = catalysis_motifs(net)["edges"]
    full = np.mean([catalysis_motifs(complementary_polymer(
        max_len=net.max_len, food_len=net.food_len, p_cat=1.0, rng=rng, **cbpm))["edges"]
        for _ in range(pilots)])
    if edges > full:
        raise ValueError(f"C-BPM reaches {full:.0f} edges at p_cat = 1; {edges} asked")
    return complementary_polymer(max_len=net.max_len, food_len=net.food_len,
                                 p_cat=edges / full, rng=rng, **cbpm)
