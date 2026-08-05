# Methodology

## Problem framing

This is a **surrogate / emulator** model, not a demand forecaster: it approximates the
outputs of expensive, manual, official analyses from their own history, so that a plausible
answer is available in seconds instead of days.

It is also **per-entity monotone regression**, not one global fit. Each field gets its own
estimator because the real relationship between price and reserves genuinely differs by field
— geology, contract terms, and development stage all differ. A single global model would
average that heterogeneity away.

The user interaction is a what-if scenario: set a market price, get back a volume and an
uncertainty band.

## Hard invariants, not hyperparameters

Four rules are enforced by construction, not tuned for:

1. **Monotonicity** — if the realized price rises, predicted volume must never fall.
2. **Hard floor** — below a field's economic abandonment threshold (breakeven), volume = 0.
3. **Re-anchoring** — at the official reference price (the most recent certified close), the
   model's prediction must reproduce the certified volume exactly (tolerance ≈ 0).
4. **Prudent ceiling** — the model never extrapolates upside beyond the largest uplift actually
   observed in training. Conservative by construction on the upside.

Any candidate method that improves an error metric but violates one of these is rejected
outright — see [`DIFFICULTIES.md`](DIFFICULTIES.md) for concrete cases where this happened.

## Why not a generic regressor

Unconstrained trees or boosting can smooth over genuine step changes (e.g. an operational
shutdown threshold) or, under a poorly specified multivariate design, invert monotonicity
entirely. The design deliberately prioritizes physical/economic admissibility over minimizing
mean error in isolation.

## The two-model chain

```text
market price  →  [Model 2]  →  realized field price  →  [Model 1]  →  reserve volume
```

| Model | Input → output | Approach |
|---|---|---|
| **Model 2** | Market price → realized field price | Captures implicit quality/transport discounts per field; robust linear regression (Theil–Sen family, with alternative families benchmarked per field) |
| **Model 1** | Realized price → reserve volume | Monotone curve per field — isotonic regression as the typical primary engine, with a smooth/PCHIP fit as an independent validation curve, and per-field overrides where justified |

Model 1 is expressed as a deviation from a baseline (the prior certified close) and then
**re-anchored**: `volume(p) = max(baseline + [f(p) − f(p_ref)], 0)`, where `p_ref` is the
Model-2 output at the official reference market price. This construction guarantees invariant
3 (re-anchoring) by algebra, not by post-hoc correction.

## Per-field method selection

There is no single algorithm applied portfolio-wide. A selection process (leave-one-year-out
and leave-one-out cross-validation, plus the hard invariants as gates) decides, per field:

- the primary Model-1 method,
- the independent validation method,
- and the Model-2 family, when an alternative improves error **without** breaking an invariant.

This research happens on an isolated `quality` track and only reaches the `production` track
after governance ratification — see [`GOVERNANCE.md`](GOVERNANCE.md).

## Why LOO-CV alone is not trusted

Classic leave-one-out cross-validation on a field with roughly 10 historical points
understates real generalization error by a factor of 4–9× relative to leave-one-*vintage*-out
(holding out an entire historical reporting period rather than a single point). Both metrics
are computed and exported; LOO-CV is treated as a fast internal proxy, not as the number that
gates production quality. The rolling backtest against real, later-arriving data (see
`GOVERNANCE.md`, gate G4) is the metric of actual pilot usefulness.

## Uncertainty as part of the product

Fields with enough historical evidence expose an uncertainty band from the quantiles of their
historical (leave-one-vintage-out) residuals. When the dashboard **aggregates** these bands to
portfolio level, the half-widths are combined in **quadrature**
(`sqrt(Σ half_width²)`), not summed. Summing implicitly assumes every field's error is
perfectly correlated with every other's, which is not a defensible assumption and produced an
unrealistically wide portfolio range in an earlier iteration — see `DIFFICULTIES.md`.

## Business reading

Because of the hard floor and per-field breakevens, lowering the market price relative to the
reference destroys more reserve volume than an equivalent price rise recovers: several
mid-size fields cross their economic floor and shut off entirely on the downside, while the
upside is capped by the prudent-ceiling invariant. The model reads as informative on the
downside and structurally conservative on the upside — that asymmetry is a property of the
domain, not a modeling artifact to correct away.
