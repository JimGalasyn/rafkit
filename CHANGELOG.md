# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Maintenance

- v0.5.0's version DOI (`10.5281/zenodo.21961131`) recorded in `CITATION.cff`. Written in
  the same commit as the DOI itself, because this commit lands *after* the release and is
  therefore the first commit of the next one — exactly where the changelog check looks, and
  where it has flagged this same gap on v0.2.0, v0.3.0 and v0.4.0.

## [0.5.0] — 2026-08-15

### Added — the firing-disk construction

- **`firing_disk_polymer`** — Serra & Villani's firing disk: a chemistry **grown** from a
  small seed rather than enumerated. A species exists only if some reaction in the network
  actually **makes** it, so the result is closed under its own production — where an
  enumerated chemistry is full of species nothing can reach.
- Reaches their published ensemble on their stated defaults: **~2046 species, ~32,000
  reactions, ~100 catalysts**, with a RAF covering >95% of the chemistry. (They report
  ~2000 species, ~40,000 reactions, *"only 100 catalysts"*, and a RAF *"often as large as
  the entire chemistry"*.)
- **`examples/serra_villani_2026_c_chemistry.py`** reproduces their Figure 3 ensemble and
  prints each statistic beside the published value, **including the one that does not
  match**: reactions per catalyst comes out ~3× their ~400, because these catalysts are
  more promiscuous (~4 catalysts per reaction against their ~1). The asymmetry against a
  K-chemistry — the published claim — is reproduced; the magnitude is not.
- ⚠ **Food is taken to be the disk.** The paper does not say what plays the role of food
  for a grown chemistry; the disk is the only externally given set. An assumption, flagged
  in the module docstring.

### Added — C-BPM, catalysis by structure

- **`complementary_polymer`** — Serra & Villani's C-BPM (*Entropy* 28(2), 184, 2026, §2.2),
  reproduced from their construction rather than invented. A catalyst carries an **active
  site**, a substring of itself, and acts on whatever is **complementary** to that site:
  cleaving any polymer containing the complement, or joining a molecule whose suffix
  complements the site's first part to one whose prefix complements its second.
- The reaction set is **identical** to the K-model's; only the catalyst→reaction assignment
  differs. That is the whole point — a K-catalyst's targets are independent draws, a
  C-catalyst's all share one template.
- Reproduces the paper's signature: at 510 species, **19 catalysts doing ~200 reactions
  each**, against the K-model's 510 catalysts doing ~18. (They report ~400 vs ~20.)
- ⚠ Implements their **total-chemistry** mode. The **firing-disk** mode is a different
  construction, not a parameter of this one, and is not implemented.

### Fixed

- **`catalysis_level` now counts the reversible pair correctly** when the two directions
  carry different catalysts. It takes the union across directions; under `paired_catalysis`
  the halves are identical so every previously reported value is unchanged. Without the fix
  a C-chemistry's cleavage catalysts were invisible to `f`.

### Added — generated inhibition

- **`binary_polymer(q=..., n_inhibitors=...)`** — inhibition can now be *generated*, not
  only hand-specified. `q` is the mirror of `p`: each (eligible molecule, reaction) pair
  becomes an inhibition edge with probability `q` (Hordijk & Steel 2012, Part II).
  Until now the u-RAF layer could only be exercised on networks written by hand, so the
  one ensemble the library actually generates could not be studied under inhibition.
- **`n_inhibitors` caps `k` exactly**, and that is the point rather than a convenience:
  `max_urafs` costs `2**k` maximal-RAF computations, so the cap is the difference between
  a feasible u-RAF census and an impossible one. `BinaryPolymerNetwork.n_inhibiting_molecules`
  reports the achieved `k`.
- Inhibition is drawn on the ligation half and **shared with its reverse** under
  `paired_catalysis`, exactly as catalysis is, so a reversible cleavage-ligation pair
  stays one unit.
- `q=0` is the default and leaves the catalysis draw **bit-identical**, so every result
  measured before inhibition existed still reproduces — asserted by a test rather than
  assumed.

### Maintenance

- v0.4.0's version DOI recorded in `CITATION.cff`.
- `docs/RELEASING.md` now points at the shared `publish-release` skill for the failure
  modes common to all five repos, and keeps only what is specific to this one — duplicated
  process drifts.
- PyPI badge URLs given a `cacheSeconds` parameter to bust GitHub Camo's cached 404 from
  before the package existed. Camo keys its cache on the URL, so only a URL change repairs
  it.

## [0.4.0] — 2026-08-15

### Added — PNML export

- **`to_pnml` / `write_pnml`** — export to PNML (ISO/IEC 15909-2), making these networks
  readable by Petri net editors, model checkers and unfolding tools. Catalysts become
  self-loops, alternative catalyst sets become separate transitions (a transition preset
  is a conjunction), and food places get source transitions so food cannot run out.
- What P/T nets cannot express is reported rather than dropped: inhibition goes in a
  `toolspecific` annotation with an in-file warning that ignoring it yields a different
  system, and reactions requiring a catalyst nothing provides are omitted and counted.

### Maintenance

- v0.3.0's version DOI recorded in `CITATION.cff` after minting.

### Fixed

- The PNML header comment could contain `--`, which is illegal inside an XML comment and
  made the whole document unparseable. The header is prepended as raw XML and so bypasses
  ElementTree's escaping; it is now sanitised. Found by validating output against an
  independent PNML reader, not by inspection — the export itself was silent.

## [0.3.0] — 2026-08-15

### Added — inhibition

- **`max_urafs`** — uninhibited RAFs (Hordijk & Steel 2012, Part II), by their
  theorem 1. Returns a **collection**: inhibition destroys the monotonicity that makes
  a maximal RAF unique, so there is no "the" maximal u-RAF, and the signature says so
  rather than a docstring.
- `is_uraf`, `is_uninhibited`, `support`, and `classes_from_inhibitors`, which groups
  by inhibiting *molecule* so that *k* — the entire cost, since the algorithm is
  `2^k` — is the number of distinct inhibitors rather than of inhibited reactions.
- `ReactionNetwork.inhibitors`, per-reaction, matching CatReNet's model.
- CRS gains CatReNet's inhibitor syntax: a brace group **after** the catalyst bracket,
  `r1 : a + b [c] {d e} -> x`, and round-trips it.

- **`simulate` respects inhibition**: a reaction with an inhibitor present has
  propensity **zero**, regardless of catalysis. Inhibition is a block where an absent
  catalyst is only a slowdown, and that difference is what lets a running network
  *lose* a subRAF rather than only gain one.
- **`irrraf_census` accepts a bare reaction set** as well as a `RafResult`, so a u-RAF
  can be passed straight in.
- **`examples/inhibition_dissolution.py`** — the static picture (two maximal u-RAFs,
  neither canonical) and the dynamic one (a subRAF that produces steadily, then stops
  dead when a rare uncatalysed event brings its inhibitor into existence).

### Unchanged, and verified to need no change

- `sample_irrraf`, `irrraf_census`, `core_raf`, `has_unique_irraf` and
  `catalytically_reachable` all already take a reaction set, and passing a u-RAF is
  correct **without further conditions**: the uninhibited property is inherited
  downward, so every sub-RAF of a u-RAF is a u-RAF. Checked on 165 sampled cores
  across seven networks, then asserted as a test rather than assumed.

### Maintenance

- v0.2.0's version DOI recorded in `CITATION.cff` after minting.

### Divergence from CatReNet, verified

- Once inhibitors are present, CatReNet's `maxRaf` filters inhibited reactions *during*
  the RAF computation, where Hordijk & Steel define an RAF without reference to
  inhibition and add the uninhibited condition separately. On a two-reaction network
  where each reaction is inhibited by the other's product, CatReNet's `maxRaf` and
  `uRaf` both return nothing while **two** maximal u-RAFs exist by the definition.
  Checked by hand against (u-1) and (u-2), and by brute-force enumeration of every
  subset on seven generated networks.

## [0.2.0] — 2026-08-15

### Added

- **`simulate`** — Gillespie direct method over a catalytic reaction network, and the
  reason it is here: a subRAF catalysed by its own products cannot start until that
  product appears by an *uncatalysed* event, so a maximal RAF does not switch on, it
  **assembles as an order-dependent sequence of rare seeding events**. The one non-unit
  constant is the uncatalysed rate reduction factor, which is the mechanism under test.
- `propensities` (mass-action with the correct `n(n-1)/2` self-pair factor) and
  `Trajectory`, which records event-resolution first-appearance times, first firing, and
  **first *uncatalysed* firing** — the seeding record.
- `simulate(..., reactions=...)` restricts which reactions may fire. This is a fidelity
  requirement rather than a convenience: the published experiment studies flow *on the
  maximal RAF*, and simulating the whole generated network instead lets species arrive by
  routes outside the set under study.

- **`core_raf` / `has_unique_irraf`** — Huson, Xavier & Steel (2024): `Core(Q)` is an
  RAF *iff* the system has exactly one irreducible RAF, and then it is that iRAF. A
  polynomial test for a question that is otherwise hard.
- **`catalytically_reachable`** — least fixpoint of what catalysed firings alone can
  make; everything outside it provably needs a spontaneous seeding event.
- **`examples/hordijk_steel_2012_seeding.py`** — the reproduction end to end, on a
  structural analogue of their network found at their published parameters (n=5, t=2,
  p=0.0045): a maximal RAF of eight two-way reactions, the same size as theirs.

### Maintenance

- v0.1.0's concept and version DOIs recorded in `CITATION.cff` after minting.

### Validated against published worked examples

- **Kauffman binary polymer example** (Huson, Xavier & Steel 2024, corollary 3.1
  illustration): the published claim is that the system is an RAF "and it contains six
  other RAFs as subsets". All seven are found exactly, and the two irreducible ones.
- **The three-iRAF system** (same paper, §4.1): all three found, none nested, and each
  contained in the union of the other two — the subtlety that makes pairwise results
  fail to extend to triples.

### Changed — catalysis is now a relation

- **`catalysts[r]` is a set of alternative catalyst *sets*** (Huson, Xavier & Steel
  2024's χ), not a flat set of molecules. Any one set being wholly present suffices;
  each set is a conjunctive requirement. This expresses two things the flat form could
  not: `{{a,d},{e}}` meaning (*a* and *d*) or *e*, and the difference between `{}`
  ("must be catalysed, nothing does") and `{frozenset()}` ("may proceed uncatalysed").
- **Constructors normalise**, so passing a plain iterable of molecules still works and
  still means what it did. Code that *reads* `net.catalysts[r]` as a flat set of
  molecules must change — use `is_catalysed(chi, available)` or
  `catalysing_molecules(chi)`.
- CRS gains brace syntax for conjunctive groups, plus `[]` and `[{}]` for the two
  empty forms.

### Validated against two further published examples, previously blocked

- **Example 3.2** (conjunctive catalysis): the four published RAFs found exactly, and
  `{r1,r2}` confirmed as the only strictly autocatalytic one.
- **The §2.4 system**: published maxRAF `{r1,r2,r5}` and unique iRAF `{r1,r2}`, with a
  test that the two empty-catalyst forms are not interchangeable.

## [0.1.0] — 2026-08-15

First release. Extracted from a private research repository with history intact.

### Added

- **RAF algorithms.** `max_raf` (maximal RAF by fixpoint, Hordijk & Steel 2004),
  `max_raf_strict` (the strictly autocatalytic variant — catalysts must be non-food
  products), `sample_irrraf` (one irreducible RAF by randomised shrinking, Steel,
  Hordijk & Smith 2012), `irrraf_census` (how many *distinct* irreducible cores a
  network carries, always a lower bound), and `is_food_catalysed`.
- **`exploitability`** — share of RAF products contributing no catalysis back, with
  `strict` / `unused` / `dispensable` conventions computed alongside one another.
- **`binary_polymer`** — Kauffman's binary polymer generator, with optional
  **cleavage** and the `paired_catalysis` convention in which a reversible
  cleavage–ligation pair is one catalysed reaction. `catalysis_level` reports *f* in
  the convention published results use.
- **`ReactionNetwork`** — arbitrary catalytic reaction systems exposing the same
  protocol the RAF algorithms use, so they work on networks that did not come from a
  polymer model.
- **CRS interchange** — `read_crs` / `parse_crs` / `to_crs` / `write_crs` for
  CatReNet's format.
- **`catrenet_strictly_autocatalytic`** — reproduces CatReNet's similarly-named
  filter, which is a *different object* from `max_raf_strict`; provided for
  interoperability and documented as such.

### Validation

- 46 tests, all hand-computed known-answer cases or fixture regressions.
- `tests/data/catrenet_polymer_n6.crs` is generated by CatReNet's own `polymer-tool`,
  and the expected counts are what `catrenet-tool` reports on it. `max_raf` agrees
  **exactly** (183 vs 183), as does `catrenet_strictly_autocatalytic` (175 vs 175).
- The binary polymer RAF phase transition reproduces the published *f* ≈ 1.20
  (Steel, Hordijk & Smith 2012, n=10, t=2) once cleavage and the paired-catalysis
  convention are both applied — 0 seeds with a RAF at *f* ≤ 1.22, all seeds by
  *f* ≈ 1.59. Without them the transition sits at *f* ≈ 4.7.

### Infrastructure

- CI on Python 3.11/3.12/3.13 with coverage to Codecov; CodeQL on a weekly cron;
  PyPI publishing by OIDC trusted publishing, guarded to tag refs only.
- `CITATION.cff`, `.zenodo.json`, `codecov.yml`, `.mailmap`, and
  [docs/RELEASING.md](docs/RELEASING.md) — the last carrying forward the release
  failure modes already paid for in a sibling repository, rather than rediscovering
  them here.

### Known divergence

- `max_raf_strict` and CatReNet's `strictlyAutocatalyticMaxRaf` compute different
  objects (161 vs 175 on the fixture). CatReNet filters the maximal RAF and does not
  re-refine, so its result need not be a RAF; `max_raf_strict` imposes the condition
  inside the fixpoint, so its result is one. Both ship, named distinctly, with the
  divergence as a test rather than a footnote.
