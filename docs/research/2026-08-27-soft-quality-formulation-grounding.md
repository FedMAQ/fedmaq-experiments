# Literature grounding for the soft-quality formulations (issue #32)

Date: 2026-08-27. Resolves the 2026-08-27 amendment to
[FedMAQ/fedmaq-experiments#32](https://github.com/FedMAQ/fedmaq-experiments/issues/32),
not the issue's original framing or the 2026-08-26 "Author's account" comment on
it — that comment argued against reformulating and is superseded by the
amendment and by [#34](https://github.com/FedMAQ/fedmaq-experiments/issues/34),
which decided 2026-08-27 to adopt the power-mean reformulation. No file outside
this document was modified. No code, config, or `chapter_3.tex` was changed.

**Location note.** No existing doc directory fit this note: `docs/adr/` holds
decision records, `docs/audits/` holds closed/archived code-fidelity audits
(distinct from a literature question), `docs/experiments/` holds run output,
and `docs/freeze/` holds frozen config snapshots. `fedmaq-analyses/docs/audits/`
was checked as the nearest analog (its dated-filename, evidence-table format is
mirrored below) but that directory is scoped to `fedmaq-analyses`, not this
repo. This document is filed under a new `docs/research/` directory in
`fedmaq-experiments`, dated like the analyses-repo audit.

## Methodology

- **In-repo corpus**: read the curated OKF layer (`fedmaq-wiki/papers/*.md`,
  `fedmaq-wiki/gaps/*.md`, `fedmaq-wiki/methods/fedmaq.md`) for the four papers
  the issue names as the natural first places to look (AdaGQ, LAQ-HC,
  AdaDQ-KD, DynFed) plus DAdaQuant, then ran a corpus-wide, case-insensitive
  grep over the raw conversion layer (`fedmaq-literature/markdown/**/paper.md`,
  all 45 papers) for `power mean|geometric mean|harmonic mean|compensat|
  Hölder|Holder|andness|orness|generalized mean`, per
  `fedmaq-literature/.agents/rules/kg-conventions.md`'s instruction to drop to
  the raw layer rather than reason only over curated summaries.
- **External search**: web search, tiered by confidence — a claim is
  "Confirmed" only when at least two independent sources agree (a search
  result plus a DOI resolution, or two independent queries), "Confirmed
  existence only" when the source's existence/title/DOI is verified but an
  internal detail (volume, pages, exact formula) rests on one AI summarizer's
  read of a paywalled page, and "Not independently verified" where flagged
  explicitly. Given this project's history with fabricated citations, no claim
  below is stated as fact without saying which tier it sits in.

---

## 1. Prior-art check (priority 1 — can change the design)

**Question:** has anyone already parameterized the *degree of compensation*
between client-side signals for adaptive quantization, precision allocation,
or resource-aware client configuration in FL?

### 1.1 In-repo corpus: no hit

The four papers the issue names, plus DAdaQuant, all combine client-side
signals — but every one uses either a single signal or a fixed linear/power-law
rule, never a continuously parameterized compensation axis:

| Paper | Signals combined | Combination rule | Compensation is tunable? |
| --- | --- | --- | --- |
| AdaGQ (Liu et al., 2023) — `fedmaq-wiki/papers/liu-2023-adagq.md` | loss-decrease rate, gradient-norm change | additive online-gradient-descent update on one network-wide average level, then a linear equalization across clients (Eq. 9–13) | No — one fixed additive rule |
| LAQ-HC (Cui et al., 2026) — `fedmaq-wiki/papers/cui-2026-laq-hc.md` | data quantity × loss, quantization-impact fit, bandwidth | `flag = (α·q + (1−α)·B) / ℓ` — a **weighted linear (arithmetic) sum** of quality and bandwidth (§3.3, Eq. cited in the wiki node) | `α` reweights two terms but the operator itself is fixed arithmetic — structurally FedMAQ's own F1, not a compensation-degree axis |
| AdaDQ-KD (Wang et al., 2026) — `fedmaq-wiki/papers/wang-2026-adadq-kd.md` | expected local delay only | precision reduced per-straggler until under a deadline threshold — single-signal, no multi-signal combination at all | N/A — only one signal drives quantization precision |
| DynFed (He et al., 2025) — `fedmaq-wiki/papers/he-2025-dynfed.md` | bit-width, prediction confidence (teacher-selection score, not the precision rule itself) | `S = α·b(x,M) + β·min p(y|x,M)` — again a **weighted linear sum** (Eq. 7–8) | No — fixed arithmetic combination, and this score selects distillation teachers, not precision |
| DAdaQuant (Hönig et al., 2022) — `fedmaq-wiki/papers/honig-2022-dadaquant.md` | one signal only: aggregation weight `w_i` | `q_i ∝ w_i^{2/3}` — a power-law allocation, but of a single signal, not a fusion of two or more with a tunable compensation exponent | N/A — one signal |

(All five paths above are relative to `fedmaq-wiki/` inside the sibling
`fedmaq-literature` repository, not this repository.)

The wiki's own gap node, `fedmaq-wiki/gaps/adaptive-precision-scheduling.md`
(authored 2026-07-10, independent of this ticket), reaches the same
conclusion from the same four sources plus DAdaQuant: *"None studies how
resource, training-state, and data-richness signals should jointly determine a
client's precision"* — corroborating evidence, not just this pass's own
reading.

A **corpus-wide grep** over the raw markdown layer (all 45 papers, not just
the five read closely) for power-mean, geometric/harmonic-mean, Hölder,
andness/orness, and compensation vocabulary found no further candidate. The
matches it did return are all false positives or unrelated senses: "stakeholder"
(substring match on "holder"), "error compensation" in the sense of residual
accumulation for biased quantizers (`mao-2023-power-load`, `cui-2026-laq-hc`,
`he-2025-dynfed`, `wang-2026-adadq-kd`, `he-2025-feddt`, `bonawitz-2019-fl-scale`,
`qin-2025-kd-survey` — all "carry forward the rounding error to the next round,"
unrelated to aggregation-operator compensation), "geometric mean" in
`hinton-2015-distillation` (the closed-form solution for pooling *teacher
predictions* under one KL direction versus the other — an ensemble-distillation
result, not a client-signal-to-precision mapping), and "harmonic mean" in
`sater-2021-anomaly-detection` (the standard F1-score definition).

### 1.2 External search: no hit, with three near-misses worth recording

- **Robust federated aggregation via generalized/power means** (e.g., γ-mean
  minimum-divergence estimation) exists and uses the same mathematical family,
  but for a different problem: robustly combining *client model updates* at
  the server against Byzantine/outlier clients, not combining *client-side
  signals* into a precision/bit-width decision. Different application of the
  same math, not prior art for this application.
- **"Multi-Criteria Client Selection and Scheduling with Fairness Guarantee for
  Federated Learning Service"** (Zhang, Zhao, Ebron, Xie, Yang, arXiv:2312.14941,
  2023) combines eleven client criteria (CPU, GPU, memory, storage, power,
  bandwidth, connection, data size, data distribution, model quality, behavior)
  into one selection score — but confirmed via the paper's own equation, this
  is `Score = w·s = Σ w_i s_i`, a **weighted linear sum**, i.e., FedMAQ's F1
  again, not a compensation-parameterized family. This is the closest external
  hit to FedMAQ's problem shape (many client-side signals, FL, resource-aware
  configuration) and it still stops at the same fixed-arithmetic-sum level as
  the in-repo corpus.
- **Multi-sensor fusion "power allocation"** hits are a false-friend: "power"
  there is radio transmit power, not the exponent of a power mean.

### 1.3 Conclusion

No prior-art hit and no novelty collision was found — in the 45-paper corpus
(both curated summaries and a raw-layer grep) or in external search across FL,
resource-allocation, and multi-criteria client-selection literature — for
parameterizing the degree of compensation between client-side signals in
adaptive quantization or precision allocation. This is an absence-of-evidence
finding, not a proof of absence: it rests on the queries actually run (§
Methodology) and could in principle be overturned by a paper using different
vocabulary that these searches did not surface. Framed against §1.1–1.2, the
accurate novelty claim is a **transplant**, not an invention: FedMAQ carries a
classical, general-purpose aggregation construction (§2) into a specific
application — FL precision allocation — where the corpus and the external
search both show prior work stopping at single-signal rules or fixed linear
sums. "We invented parameterized compensation" would be false (§2 shows it
predates FedMAQ by decades in MCDM); "we are the first to apply a
compensation-parameterized aggregator to client-signal-driven precision
allocation in FL, where existing work uses only single-signal or fixed-weighted-
sum combination" is the claim the evidence supports.

---

## 2. Citable grounding for the power-mean family (priority 2)

### 2.1 The family and its limits — confirmed

**Hardy, G. H.; Littlewood, J. E.; Pólya, G.** *Inequalities.* Cambridge
University Press, 1934 (2nd ed. 1952). **Confirmed**: existence, authors, title,
publisher, and both edition years cross-verified via a contemporary *Mathematical
Gazette* review of the 1934 edition, a 1935 *Science* journal review, the 1952
second-edition review, and current publisher/bookseller listings (Cambridge
Core, AbeBooks, Google Books). This is the standard reference for the weighted
power mean `M_p` and the theorem that `M_p` is non-decreasing in `p`, with `p→0`
giving the weighted geometric mean and `p→−∞` giving the minimum — the specific
chapter/section numbering is stated here from general mathematical knowledge,
not verified against the primary text directly in this pass, so cite the
theorem, not a specific page, without a direct check.

**Bullen, P. S.** *Handbook of Means and Their Inequalities.* Mathematics and
its Applications, vol. 560. Kluwer Academic Publishers, Dordrecht, 2003.
**Confirmed** via the Springer re-listing (DOI-bearing, `10.1007/978-94-017-0399-4`)
and independent bookseller/library listings; has a dedicated "The Power Means"
chapter. A modern, more accessible companion to Hardy–Littlewood–Pólya for the
same family — worth citing alongside it rather than instead of it, since it is
the more recent restatement.

### 2.2 The compensation axis — confirmed, but not power-mean-shaped on its own

**Zimmermann, H.-J.; Zysno, P.** "Latent connectives in human decision making."
*Fuzzy Sets and Systems*, 4, 37–51, 1980. **Confirmed**: title, authors,
journal, volume, and page range cross-verified across independent citing
sources (ResearchGate abstract, SciRP reference listing). This is the
foundational source for "aggregation with a continuously tunable degree of
compensation" in the compensatory/non-compensatory sense `chapter_3.tex:164`
and `:170` already describe without naming: it introduces a `γ`-operator that
interpolates between the (non-compensatory) intersection/min and the
(compensatory) union/product with an explicit compensation parameter, based on
experimental evidence that human judgment uses partial compensation rather
than pure logical AND/OR.

**This source grounds the compensation *concept*, not the power-mean
*construction*.** The Zimmermann–Zysno `γ`-operator is a product/algebraic-sum
hybrid, not a weighted power mean — citing it for "the exponent `p` of a power
mean controls compensation" would be exactly the kind of adjacent-authority
misattribution this ticket's warning is about. The bridge between the two is
§2.3.

### 2.3 The bridge: power-mean exponent *as* the compensation parameter — confirmed

**Dujmović, J. J.** "Weighted conjunctive and disjunctive means and their
application in system evaluation," 1974. **Confirmed existence, author, title,
and year** — this combination appears identically across multiple independent
search results citing it (consistent with a Springer chapter's, an arXiv
survey's, and a ScienceDirect article's reference lists all pointing to the
same work), strong convergent evidence for a pre-digital-era reference with no
single canonical online host. **Not independently verified in this pass**: the
publication venue/series name, issue number, and page range that appeared in
one search synthesis ("Zbornik Radova," Univ. Beograd Publ. Elektrotehn. Fak.,
no. 483, pp. 147–158) were not seen directly in any tool result and should be
checked against a library catalog or a citing paper's actual reference list
before going into the bibliography — do not copy those numbers from this note
without that check. This is Dujmović's original definition of the weighted
power mean used as a **generator function** for a continuous family running
from full conjunction (`p→−∞`, non-compensatory) through the arithmetic mean
(`p=1`) to full disjunction (`p→+∞`), with the exponent `p` (his "andness/orness"
parameter) governing exactly the compensation degree, and with a documented
worked example (andness `0.8125 ↔ p = −1.6544`) confirming the parameter is
continuous, not a small enumerated set.

**Dujmović, J. J.; Larsen, H. L.** "Generalized conjunction/disjunction."
*International Journal of Approximate Reasoning*, 2007. **Confirmed existence
only**: title, authors, and DOI (`10.1016/j.ijar.2006.12.011`, resolving to
ScienceDirect PII `S0888613X07000035`) are consistent across queries. **Not
independently verified**: two search passes returned conflicting volume/pages
for this article — "46(3), 423–446" and "47(3), 405–423" — a single-summarizer
inconsistency of exactly the kind this project has been burned by before.
Cite by DOI; verify the print volume/page range directly against the publisher
page (blocked by a 403 in this pass) or a library copy before it goes in the
bibliography.

This pair is the direct citation for the manuscript's central claim: that the
power-mean exponent `p` is not merely a mathematical curiosity but *is*, in an
established literature, the operationalization of "degree of compensation."
Framed honestly: **HLP/Bullen ground the family and its limits; Zimmermann–Zysno
ground the compensation concept; Dujmović is the source that fuses the two** —
the power mean used specifically as a compensation-degree control. Any
manuscript sentence identifying `p` with compensation should cite Dujmović (and
may cite Zimmermann–Zysno alongside it for the concept's older, non-power-mean
origin), not Hardy–Littlewood–Pólya alone.

### 2.4 A secondary MCDM example — confirmed existence, form not verified

**Aggarwal, M.** "Compensative weighted averaging aggregation operators."
*Applied Soft Computing*, 2015. **Confirmed existence and venue**: a DOI lookup
(`10.1016/j.asoc.2014.09.049`, found via a dblp search) redirects to
ScienceDirect PII `S1568494614005419`, the same article a separate title
search returned — two independent paths converging on the same article.
**Not independently verified**: the specific volume/page ("28, 368–378") comes
from one AI summarizer's read of the same page and a direct fetch of the
journal's volume-28 table of contents did not list a matching entry, so that
detail is flagged rather than asserted. This pass did not access the article's
full text, so the exact operator formula is not described here — only that
independent secondary sources describe it as a weighted-averaging family with
"a dedicated parameter to model compensation." Useful as a second, more recent
MCDM data point that the compensation-as-parameter idea is an active research
line, but Dujmović (§2.3) is the stronger and better-verified citation for the
manuscript's actual claim.

### 2.5 Related, not required

**Yager, R. R.** "On ordered weighted averaging aggregation operators in
multicriteria decisionmaking." *IEEE Transactions on Systems, Man, and
Cybernetics*, 1988. **Confirmed existence, author, title, year, and venue**
(an ACM-indexed search result carried these). **Not independently verified**:
volume/issue/page numbers ("18(1), 183–190") are from general knowledge, not a
tool result in this pass — check before citing precisely. A
different aggregation family (order-statistic weights, not a power exponent)
that also spans AND-like to OR-like behavior via an "orness" measure. Mentioned
for completeness; not needed to support the power-mean claim and not required
reading for §3.3.

---

## 3. What survives from the original scope: F3 and F4 (priority 3)

The amendment keeps F3 (priority-with-modulation) and F4 (thresholds) as still
uncited under #34's design. Given the amendment's own priority order, this
section received comparatively less search effort; treat it as a starting
point, not a closed question.

**F3** (`s = g̃·(1+κñ)/(1+κ)`, gradient-primary with a bounded data-richness
modulator) is structurally adjacent to a body of work on **prioritized
aggregation operators** in MCDM, where lower-priority criteria are reweighted
by how well higher-priority ones are already satisfied. A search result
referenced work under this name by Yager, but the result that came back was a
*different* paper (a related title in the *International Journal of
Intelligent Systems*, different authors) than the "Prioritized aggregation
operators" title expected from general knowledge — so **no specific citation
for this family is confirmed in this pass**; the existence of a "prioritized
aggregation operators" research line is plausible from general knowledge but
was not independently verified here and should not be cited from this note.
What is confirmed is only the structural observation: F3's asymmetric,
one-signal-leads shape is a recognized category in MCDM, distinct from both the
compensatory/non-compensatory axis and from Dujmović's GCD family. No source
found in this pass states F3's exact bounded-multiplier form. Best-supported
honest framing: **deliberately original**, structurally in the family of
priority/hierarchical aggregation but not confirmed as an instance of any
specific named operator.

**F4** (three-tier AND/OR/otherwise threshold rule) is in the lineage of
**conjunctive/non-compensatory choice rules**. **Tversky, A.** "Elimination by
aspects: A theory of choice," 1972. **Confirmed existence, author, title, and
year** (search results carried these plus a citation count in the thousands,
independently confirming this is an extremely well-established source and the
classical origin of threshold-based, non-compensatory multi-attribute choice
rules). **Not independently verified in this pass**: the specific journal,
volume, issue, and page range (general knowledge suggests *Psychological
Review*) did not appear in any tool result and must be checked directly before
citing precisely. F4's exact three-tier AND/OR structure is, in any case, not
identical to Tversky's iterative elimination process, so this citation would
support the conjunctive/threshold *tradition* F4 belongs to, not F4's specific
rule. A possibly-earlier and more exact match — Dawes, R. M. (1964), on
conjunctive/disjunctive screening rules — was not searched in this pass at all
and should not be cited without first confirming it exists.

---

## 4. Corrections to existing manuscript claims found during this pass

Two findings surfaced while reading `chapter_3.tex:149–190` for context, not
asked for but load-bearing for whatever text replaces this section under #34:

- **`chapter_3.tex:170`** states F2 (the multiplicative formulation) means "no
  compensation is possible." Per §2.1's own definitions, this is an
  overstatement carried over from the superseded four-formulation framing: the
  **weighted geometric mean is partially compensatory** — strictly between the
  fully compensatory arithmetic mean and the fully non-compensatory minimum,
  not at the non-compensatory extreme itself. This matches #34's own
  compensation table, which correctly lists F2 as "partially compensatory,"
  not "non-compensatory" — the superseded 2026-08-26 comment on this issue
  used "non-compensatory" for F2, which is the less accurate framing.
- **F2's actual formula**, `s = g̃^{ω₁}·ñ^{ω₂}`, has **unconstrained** exponents
  in `chapter_3.tex:166–170` — it equals the weighted geometric mean only on
  the simplex `ω₁+ω₂=1`, which the evaluated point `ω₁=ω₂=0.5` satisfies but
  which the signal-removal ablation arms deliberately leave (using `x^0=1` to
  retire a factor). #34's D4 already proposes retiring this convention in
  favor of `ω=0`/`ω=1` under the power mean, which resolves the imprecision
  this note is flagging.

---

## Verification ledger

| Claim | Tier |
| --- | --- |
| No FL-corpus or external prior art for compensation-parameterized multi-signal precision allocation | Absence-of-evidence, methodology stated above |
| AdaGQ / LAQ-HC / AdaDQ-KD / DynFed / DAdaQuant combination mechanisms | Confirmed — read directly from curated wiki nodes and, for the manuscript-cited claims, cross-checked against `chapter_3.tex` |
| Hardy–Littlewood–Pólya, *Inequalities*, 1934/1952 | Confirmed (multiple independent reviews/listings); internal chapter/section not directly checked |
| Bullen, *Handbook of Means and Their Inequalities*, 2003 | Confirmed |
| Zimmermann & Zysno, "Latent Connectives in Human Decision Making," 1980 | Confirmed |
| Dujmović, "Weighted conjunctive and disjunctive means...," 1974 | Confirmed: author/title/year (convergent independent citations). Venue name, issue number, page range NOT independently verified — do not copy those numbers without checking a library catalog or citing paper's reference list |
| Dujmović & Larsen, "Generalized conjunction/disjunction," 2007 | Confirmed existence/DOI/authors/title; volume/pages inconsistent across sources — verify before citing |
| Aggarwal, "Compensative weighted averaging aggregation operators," 2015 | Confirmed existence/DOI; volume/pages and operator formula not independently verified |
| Yager, OWA operators, 1988 | Confirmed author/title/year/venue; volume/issue/pages NOT independently verified |
| "Prioritized aggregation operators" (Yager) | NOT confirmed — the search result returned a different paper than expected; do not cite from this note |
| Tversky, "Elimination by aspects," 1972 | Confirmed author/title/year (plus citation-count evidence of prominence); journal/volume/issue/pages NOT independently verified; exact match to F4's specific rule not confirmed |
| Dawes (1964) as an earlier F4 source | Not searched at all in this pass — do not cite |
