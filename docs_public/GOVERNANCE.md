# Governance

## Two tracks: production and quality

The pipeline runs in one of two isolated tracks, selected by an environment flag:

- **`production`** — the track that feeds the published dashboard. Conservative by default;
  new methods and flags start **off** here.
- **`quality`** — a fully isolated research track (its own staging directory, its own results
  directory, its own test suite) where new engines, features, and selection rules are tried
  and measured before anyone considers promoting them.

Both tracks run the **same rigor**: the same per-phase pytest gates and the same non-regression
contract, so "it only broke in quality" is never used to excuse a hidden bug — a bug that
`quality` was once allowed to skip past pytest is exactly what motivated tightening this rule
(see `DIFFICULTIES.md`).

Promotion from `quality` to `production` happens only after the change is measured, documented
with its rationale, and ratified — not automatically on a passing test run.

## The non-regression contract ("NORTH")

A fixed subset of the portfolio — the fields that concentrate the largest share of modeled
volume — is frozen as the golden gate. Every pipeline run is checked against five gate
families:

| Gate | Checks | Blocking? |
|---|---|---|
| **G1 — Physical** | Monotonicity (0 violations), sub-abandonment volume ≈ 0, upper asymptote within a sane multiple of historical max | Yes, all fields |
| **G2 — Statistical** | Minimum evidence count per golden field; skill above the naive baseline; bounded relative error (primary < 20%, validation < 40%) | Yes, golden fields, with documented exemptions for structurally flat fields |
| **G3 — Non-regression vs. frozen baseline** | Error must not worsen by more than ~10% relative to the last accepted baseline, nor lose more than 0.05 of skill | Yes |
| **G4 — Rolling backtest** | When a predicted quarter's real data later arrives, compare the prediction actually made against the real outcome, per field | Manual, quarterly cadence — this is the actual measure of pilot usefulness, not a proxy |
| **G5 — Label & Model-2 stability** | Confidence-label distribution and Model-2 fallback count don't drift more than a small tolerance vs. baseline | Yes, reporting-level |

Updating the frozen baseline is itself governed: it requires a written justification (what
changed and why it's an improvement) recorded alongside the change, and the new baseline is
committed in the same unit of work — never silently.

## What is deliberately *not* a gating criterion

- Raw R² from LOO-CV — it is structurally negative when the model is evaluated inside a narrow
  historical price band (see `DIFFICULTIES.md`); reporting it as a KPI would penalize a model
  for the band being narrow, not for being wrong.
- Metrics computed on the synthetic anchor points — those points encode domain knowledge
  (breakeven floors), not evidence to be cross-validated against.
- Divergence between the primary and validation engines outside the historically observed
  price band — extrapolation disagreement there is expected, not a defect.

## Rolling retraining discipline

The pilot's intended cadence is: predict the next reporting period, label that prediction
explicitly as a forecast, snapshot it immutably, and — critically — **retrain only on real,
audited data once it arrives**. Saved predictions never re-enter training as if they were
observations, even after the period they predicted has passed. This is what makes the G4
rolling backtest meaningful: the model that gets graded against reality is never the one that
was shown the answer in advance.
