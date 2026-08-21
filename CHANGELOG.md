# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- **`dg_assoc` is now required, with no default**, in every `rafkit.thermo` entry point whose
  answer depends on it: `reaction_free_energies`, `reaction_rate_constants`,
  `detailed_balance_residual`, `kinetics_from_energies`, `transfer_matrix`, `elongation_ratio`
  and `mean_length`. ⚠ **Breaking**, and deliberately so: `0.0` is not a neutral absence but the
  claim *"joining two molecules is free at the standard state"*. Ross & Deamer (*Life* 6(3):28)
  put phosphodiester formation at +3.3 kcal/mol at 85 C (~ +4.7 RT, K1 ~ 1e-3), so zero is not a
  small value of this quantity — it is a different claim. It also decides whether an equilibrium
  exists: at monomer 0.5 with `E = -1`, `dg_assoc = 0` puts the elongation ratio above 1 (runaway,
  no equilibrium) and +4.7 brings it back below. Same rule as `permeation_flux` requiring both
  volumes.

  ⚠ It is also where **water activity** enters — a ligation releases water, so mass action puts
  `+RT*ln(a_W)` in exactly this term. The old default silently asserted `a_W = 1`, permanently wet.
  Not implemented; that is the form, not a number.

  ⚠ **One deliberate exemption**: `sequence_correlation_length` keeps its default, because there
  the argument *provably cancels* (asserted in the suite at `dg_assoc=7.0`). Requiring a value
  that cannot change the answer would train the reflex of typing `dg_assoc=0.0` without thinking,
  which is exactly what makes the requirement worthless everywhere it matters.

  Updating the suite made the scale of the old silence visible: **32 call sites** were relying on
  the default, i.e. asserting `a_W = 1` without saying so.

### Added

- **`rafkit.thermo` reaches the simulator.** `Kinetics` (per-reaction *uncatalysed* rate
  constants plus the factor a **present** catalyst applies), `kinetics_from_energies`, and
  `propensities(..., kinetics=...)` / `simulate(..., kinetics=...)`. The split keeps the
  enhancement a ratio: `thermo` says what the rate constants are, presence is state, and the
  simulator never holds a catalysed rate constant it could apply to one direction. The default
  path is untouched — `kinetics=None` runs the same code as before.

  ⚠ The existing model was **already in this form**: `gillespie`'s uncatalysed factor of 20 is
  `Kinetics.uniform(net.n_reactions, 1/20, 20)` and reproduces its propensities to the last ulp (they
  differ by 2.2e-16 on 16 of 136 reactions, because `1/20` is not exact in binary). So catalysis
  was never the missing piece — the *equilibrium* was. Unit constants make `k_f = k_r` whether or
  not a catalyst is present, so `K_eq = 1` regardless.

  What changes a trajectory: a chemistry built from bond energies has its **stationary point at
  the thermodynamic equilibrium**, and a present catalyst raises both directions by the
  enhancement without moving it.

  ⚠ New: `unpaired_catalysis`. `binary_polymer(paired_catalysis=False)` is not a variant
  chemistry but a **thermodynamically impossible** one — a molecule catalysing a ligation but not
  its cleavage is a free-energy source whenever it is present, measured at 100x net flux from
  nothing on a hand-built case. `kinetics_from_energies` refuses such a network, because lawful
  rate constants on an unlawful catalysis assignment are still unlawful and the rates alone
  cannot see it.

  ⚠ A `kinetics` run is still **driven**: the food floor holds a chemical potential at the
  boundary, so it does not relax to the equilibrium ensemble `thermo` computes.

  ⚠ Seven findings from code review, all reproduced first. (a) The canonical example quoted in
  five places, `Kinetics(k_uncat=1/20, enhancement=20)`, **raised `ValueError`** — `k_uncat` is
  per reaction and a bare scalar cannot say how many; added `Kinetics.uniform(n, k, e)` and
  corrected all five. (b) The headline balance invariant `n_ab/(n_a*n_b) = K` is **false for
  self-ligation**: `a + a -> aa` takes `n_a(n_a-1)/2`, so the count ratio tends to `K/2` — at
  `n_a = 20` under `K = 1`, balance is at 190, not 400. The map from ΔG to a count ratio is not
  uniform across the chemistry, and the test suite already sidestepped `a == b` without the
  docstring saying so. (c) `Kinetics` checked positivity but not finiteness: `inf` passed, then
  made every sampling probability NaN. (d) The mutual-exclusivity guard used the live default as
  its own sentinel, so an explicit `uncatalysed_factor=20.0` alongside `kinetics` was silently
  accepted and `simulate`'s forwarding worked only by that coincidence; now a `None` sentinel.
  (e) Inhibitor pairing was unchecked — inhibition is an absolute block, so a one-sided inhibitor
  makes one direction impossible while the other fires; added `unpaired_inhibition`. (f)
  `propensities` re-derived the enhancement rule instead of calling `Kinetics.rates`, so the
  method built for the simulator was never exercised by it. (g) `reversible_pairs` ran three
  times per `kinetics_from_energies`; now once, threaded — and the test written to pin that
  caught a fourth call site the refactor had missed.

- `rafkit.thermo` — free energy for a polymer chemistry: bond energies, the rate constants
  they force, and the equilibrium ensemble they induce. **Deliberately off-theme** like
  `dilution` and `permeation`, and here because the rest of the library already has an
  *implicit* thermodynamics and it is the wrong one. `gillespie` gives every reaction a unit
  rate constant, so in a cleavage–ligation chemistry `k_f = k_r`, `K_eq = 1` and `ΔG° = 0`:
  every polymer is asserted isoenergetic with its parts. Measured, that scores a
  `detailed_balance_residual` of 2.5 against bond energies of −1 to −3 with an association
  cost of 0.5, where rates built from those energies score 2e-16.

  Catalysis becomes a **ratio applied to both directions**, which is the only form that
  leaves `K_eq` alone. Enablement (`k_uncat = 0`) scores `inf` — a catalyst that decides
  whether a reaction exists also moves its equilibrium, from unreachable to reachable, and
  no catalyst does that. Detailed balance is structural, not imposed: the barrier is split
  in the Brønsted way, so `k_f/k_r = exp(-ΔG/RT)` identically for every barrier and every
  `beta`, and `k_uncat` becomes a consequence of the barrier rather than a switch.

  Sharpens the design premise it was built from. "A uniform bond energy admits no sequence
  preference" is true but not tight: the ensemble is a 1-D Ising chain, and its second
  transfer eigenvalue vanishes for **every additive** assignment `E00 + E11 = E01 + E10`, not
  only the uniform one. So `nonadditivity` `ε = E00 + E11 - E01 - E10` is the whole of the
  sequence preference in one number — `ε > 0` alternating, `ε < 0` blocky, `ε = 0` blind —
  and three energies chosen additively buy nothing over one.

  ⚠ Calibration tier is **algebraic**: identities that hold or do not, reproducing no
  experiment and calibrating against no published number. It says the model is consistent,
  not that it is right.

  ⚠ Four findings from code review, three of them silent wrong answers. (a)
  `nonadditivity` used `2*E01` where the condition is `E01 + E10`, so it reported a sequence
  preference of -3 for a genuinely additive non-symmetric assignment and contradicted
  `sequence_correlation_length` on the same object — the matrix is not required to be
  symmetric, and the doubled form is right only when it is. (b) `reaction_rate_constants`
  checked `enhancement` across reversible pairs but not `barrier`, `beta` or `prefactor`, so
  a per-reaction barrier gave the two halves of one reversible reaction different transition
  states and passed, at a residual of 4.0. Not a catalyst, and just as much a free-energy
  source; every per-reaction quantity is now checked by one helper. (c) `rate_constants`
  crashed on the array `beta` it documents as supported (`truth value of an array is
  ambiguous`). (d) `detailed_balance_residual` indexed `k` without a length check: a bare
  `IndexError` when short, a residual over the wrong reactions when merely misaligned.

  ⚠ Three corrections found while testing, the first two recorded because the null case hid
  them. (i)
  An additive assignment returns a sequence correlation length of 0.027 bonds rather than 0
  — roundoff in `exp` reported as sequence memory — so subdominant eigenvalues below 1e-12
  of the leading one are read as zero, and the 2×2 case is taken in closed form where
  `λ₂ = 0` is exact. (ii) The geometric length distribution starts at the first **bond**,
  not the first molecule: a monomer has no bonds, so the step from length 1 to 2 is a
  boundary term. A uniform assignment happens to satisfy it anyway, which is exactly why it
  went unnoticed; on an additive-but-not-uniform assignment at `ρ = 0.4` the first step is
  0.377 and `mean_length` overstates the number-average by 1.4%. (iii) Perron–Frobenius
  constrains the **leading** eigenvalue and nothing else: a positive matrix of size 3 or more
  may have complex subdominant eigenvalues, and a randomly drawn 4x4 bond-energy matrix does
  so on the first try. Rejecting them as impossible refused an ordinary nucleotide chemistry;
  they are legitimate — a complex pair means a correlation that oscillates as it decays — and
  the decay length is set by the modulus either way.

- `rafkit.permeation` — size-selective transport across a compartment membrane, following
  Hordijk, Naylor, Krasnogor & Fellermann (*Life* 8(3), 33, 2018). **Deliberately off-theme**
  like `dilution`: it takes no `ReactionNetwork`. It is here because "permeation is
  proportional to the **concentration** difference" is not "proportional to the **count**
  difference" — the two coincide only when compartment and medium have equal volume, and in a
  spatial model they generally do not (radius-0.5 sphere in a 2.5x2.5x1 voxel is a ratio of
  11.9). Writing the flux as `P*(n_out - n_in)` produces plausible numbers rather than an
  error. Reproducing that paper's induction experiment from the authors' published input
  files, the count form cannot match both published arms at any permeability, reaching the
  control value at an effect ratio of 1.15 or 2.44 and bracketing the published 1.60; the
  concentration form reproduces both. Calibration tier is a published *figure* with one
  fitted parameter — not a second analytic anchor.

- `rafkit.dilution` — serial-dilution and CSTR protocols for competing autocatalytic sets,
  reproducing the minimal model of Matsubara, Ameta, Thutupalli, Nghe & Krishna
  (arXiv:2211.03155). **Deliberately off-theme:** it takes no `ReactionNetwork` and has no
  RAF structure, and the README says so. It is here because it is the library's only
  *analytic* calibration — every other check is against a reference implementation or a
  published figure, whereas these are closed-form conditions reproduced to 2e-9. Relevant
  because RAF work increasingly runs networks inside growing, dividing compartments, where
  the dilution protocol is a modelling choice that changes the answer.

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
