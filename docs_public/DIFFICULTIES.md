# Difficulties, findings, and decisions that didn't make the cut

This is the honest record: what went wrong, what was tried and rejected, and why. It's the
part of the project most worth reading if you want to know what was actually hard, as opposed
to what shipped.

## The flagship finding: price effect confounded with recertification

For a meaningful number of fields, what an early version of the model attributed entirely to
"price sensitivity" was actually a mix of two different things:

1. genuine price → volume response, and
2. **year-to-year recertification jumps** — the official registry re-books a field's reserves
   between one reporting year and the next for reasons unrelated to price (new wells, revised
   engineering estimates, rule changes).

The failure mode: when one year's observed price band and the next year's don't overlap at
all, there is no within-price variation left to estimate a clean price slope — the year-over-year
step gets absorbed into what looks like elasticity. A model trained on that data literally
cannot tell the two apart from the data alone.

**Diagnosis** used three signals together: how much of a field's delta variance is explained by
reporting-year alone (should be low if price is doing the work), how much the years' price
bands overlap (should be wide if price is separable from year), and whether the resulting
year-over-year jump is large enough to matter financially. Where price sensitivity was not
separable in-band, the field is now labeled as such rather than silently kept.

**What was done about it**: the confound was measured (not just asserted), a governance label
was added so it's visible in the product rather than buried in a diagnostic file, confidence
was capped where the confound was material, and the underlying volume-normalization logic was
corrected to compare against the closing baseline of the prior year rather than the same year
— which is what let the confound get read as a price signal in the first place.

**The transferable lesson**: identifiability beats adding more hyperparameters. A better error
metric does not, by itself, justify a causal story the data cannot actually support. A visible
label describing *why* a number should be trusted less is part of the product, not an internal
appendix.

## The negative-R² band is real and expected — not a bug

Cross-validated R² on several fields is structurally negative when computed over the
historically observed price band, which spans roughly one to two decades of dollars. That is
not a sign of a broken model — R² measures skill relative to predicting the *mean* of the
target, and over a narrow-enough input range with real dispersion, a naive mean predictor can
outperform a fitted curve on held-out folds by pure variance, even when the curve's shape is
correct in a wider sense. R² was formally excluded from the non-regression gate (see
`GOVERNANCE.md`) for exactly this reason; the primary blocking statistic is relative error
against a naive baseline instead.

## An experiment that was run and rejected: recency weighting

The intuitive fix for "the newest data point should matter more" — weighting the most recent
reporting vintage 2× or 3× in the fit — was implemented, benchmarked, and **rejected**: with
roughly 9 real observations per field, upweighting the newest one amplified noise rather than
signal and measurably worsened the cross-vintage error metric. Uniform weighting across
vintages was kept. This is included specifically because it's a plausible-sounding idea that
the data said no to, and that's worth recording as much as the ideas that worked.

## A bug that a skipped test suite let through

An early version of the volume-floor logic could produce a **negative** predicted volume for
one field under a specific combination of historical-data-driven output caps. It was invisible
for a period because the research track was, at the time, allowed to skip the full pytest
suite that would have caught it. The fix was a straightforward hard-zero clamp; the actual
lesson was procedural — the two tracks now run identical test rigor (see `GOVERNANCE.md`),
specifically so "it's just the research track" stops being a reason a defect survives.

## Shape overrides that improved MAE were still rejected

Several alternative curve shapes for Model 1 (sigmoid, piecewise-linear, flat) beat the
default isotonic engine on cross-validated error for some fields. They were adopted only where
they didn't change the *semantic* meaning of the curve for that field's actual physical
behavior (e.g., a genuinely flat deck with no measurable price sensitivity). Where a shape
swap would have implied a different physical story than the field's evidence supported, it was
rejected even though the metric improved — this project treats "the curve looks like the
domain" as a constraint, not a tie-breaker.

## Why portfolio bands are quadrature, not sum

An earlier version of the aggregated uncertainty band simply summed each field's half-width to
get the portfolio-level band. That silently assumes every field's error moves in lockstep,
which produced a portfolio band wide enough to be useless (on the order of several times too
wide relative to what the individual field bands would imply under a defensible independence
assumption). Switching to quadrature combination (`sqrt(Σ half_width²)`) narrowed the reported
band to something the business could actually act on, and is now the standard aggregation
rule — see `METHODOLOGY.md`.

## Data engineering pitfalls worth naming

- **Identity resolution across systems**: the same field is spelled, cased, or split
  differently across the source systems that feed the pipeline, and that mapping changes over
  time as fields get administratively merged. It is resolved through an explicit business
  dimension table maintained by domain owners, not through code-level string matching — the
  one time an ad-hoc override was added directly in code, it produced a stale ghost value that
  took a session to track down.
- **Component aggregation vs. "pick one"**: composite fields whose physical sub-parts coexist
  in the source data were originally deduplicated by keeping the first record seen — which
  silently discarded real volume from the other components. Fixed to sum the components and
  volume-weight the blended price.
- **Locale-sensitive numeric parsing**: one major source file uses comma as the decimal
  separator in one sheet and a period in another. Reading both with the same parser silently
  produces garbage in one of them if the locale assumption isn't made explicit per-column.
