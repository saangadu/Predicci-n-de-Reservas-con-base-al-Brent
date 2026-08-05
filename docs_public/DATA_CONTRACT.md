# Data contract

This describes the **shape** of the three input workbooks the pipeline expects — sheet names,
header row layout, column names and types, and known parsing traps — with **no real values**.
None of these files are in this repository (`datos/raw/` is gitignored). To run the pipeline
against a different portfolio, supply your own workbooks matching this contract, or adapt
`01_etl.py`'s readers to your own source shape.

## 1. Certified reserves history workbook

One sheet holding the audited annual reserves close per field, plus a certified sensitivity
matrix.

| Aspect | Detail |
|---|---|
| Reserves sheet | Long format: one row per (field, year); reserve-class columns (proved developed producing / proved non-producing / proved undeveloped) |
| Decimal convention | **Comma** decimal separator (e.g. `"0,000"`) — must be parsed as `str.replace(',', '.').astype(float)`, not read as a locale-aware float directly |
| Price columns (same workbook) | **Period** decimal separator — read directly, do not apply the comma fix here |
| Sensitivity matrix sheet | Header row is **not row 1** — real header lives 2 rows down (`skiprows=2` in the reader); an "Activo" (asset) identifier column and a "CAMPO" (field name) column anchor each row |
| Encoding | Must be read with `engine='openpyxl'`, not the default engine, to preserve accented characters in field names |

## 2. Quarterly consolidated deck workbook

A wide-format workbook with one sheet per reporting quarter (e.g. `2024_Q1` … latest) plus a
`Consolidado` sheet that concatenates all quarters column-wise.

| Aspect | Detail |
|---|---|
| Layout | **Variable-width blocks** per metric (net price, quality discount, transport discount, reserves), not fixed column offsets — the reader must detect each block's start by matching its header label, not by hardcoded column index |
| Header pattern | Column titles follow `"<Metric name> <YYYY>_Q<N>"` (e.g. `Net Price 2025_Q3`); a regex extracts the metric and the quarter from the label |
| Quarter identifier | String format `YYYY_Qn`, used throughout the pipeline as the vintage key |
| Per-quarter sheets | Carry price columns; the reserves/sensitivity column for a not-yet-closed quarter may exist as a placeholder before real data arrives — the pipeline does not assume every quarter sheet has real reserves just because the column exists |
| Field key | Matched to the unified field dimension (see below), not used as a raw string key |

## 3. Field economic-threshold ("breakeven") workbooks

One workbook per field per certification vintage, holding a cash-flow projection used to
derive two economic threshold prices per field: the price at which the field's economic limit
is reached, and the price at which net present value crosses zero.

| Aspect | Detail |
|---|---|
| File naming | `<FIELD>_CF_SEC_<date>.xlsx`, field name extracted via regex `^(.+?)_CF_SEC_.*` |
| Content | Multi-year cash-flow line items (revenue, opex, capex, net oil volume) per field, per vintage |
| Vintage | One workbook can supersede an older one for the same field; the pipeline always uses the most recent available vintage per field |

## 4. The field identity dimension

A separate small reference table (not one of the three above) mapping every raw field-name
spelling encountered across the three sources to one canonical "unified" field key. This table
is maintained by the domain owner, not derived from the code — merges, renames, and casing
fixes are corrected in this table, never by adding string-matching exceptions in the pipeline.

## What the pipeline produces internally (not published)

The ETL phase (`01_etl.py`) resolves the three sources above against the field dimension into
one unified long table (one row per field × vintage × scenario), which every later phase reads
from. Its schema is documented in the internal (gitignored) `docs/MAESTRO.md` — it is not
reproduced here because several of its columns carry real figures once populated.
