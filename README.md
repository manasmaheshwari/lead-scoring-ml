# lead-scoring-ml

A small, end-to-end lead scoring project: a synthetic lead generator, a gradient-boosting (or logistic regression) conversion model, a scoring CLI, and SHAP explanations for *why* a lead scored the way it did.

## Why

Sales teams can't call every lead. A score is only useful if reps trust it and managers can reason about the capacity trade-off. This repo shows the full loop: predict, rank, explain, monitor. All data is synthetic, so it runs anywhere with no confidential inputs.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m leadscore train                       # prints test AUC + top SHAP drivers
python -m leadscore generate --n 200 --out data/new_leads.csv
python -m leadscore score data/new_leads.csv --top 10
python -m leadscore explain data/new_leads.csv --row 0
pytest
```

`python -m leadscore train --model logreg` swaps in a logistic regression baseline.

## The data

Features: `source`, `company_size`, `pages_viewed`, `demo_requested`, `days_since_first_touch`, `emails_opened`. The label `converted` is drawn from a known logistic function plus noise, so SHAP drivers can be checked against ground truth (demo requests and referrals should dominate).

## Project structure

```
leadscore/
  data.py     seeded synthetic generator
  model.py    pipeline, train/AUC, score, SHAP global + per-lead
  cli.py      generate | train | score | explain
tests/        generator, AUC, score range, SHAP sanity, CLI end-to-end
```

## PM notes

**How sales would use the score.** Rank the daily queue by score. Route the top band to reps for same-day calls, the middle band to sequenced email/nurture, and the bottom band to automation only. Per-lead SHAP output becomes a one-line "why this lead" hint in the CRM (for example "requested a demo, referral source") so reps trust and act on it instead of treating it as a black box.

**Threshold vs capacity.** Pick the threshold from capacity, not in the abstract. If a team can work 200 leads a day, the threshold is the score of the 200th lead. Raising it improves precision (fewer wasted calls) but lowers recall (converting leads left unworked). Judge the model by precision@k and lift over the base rate, not AUC alone: AUC says the ranking is good overall, capacity decides which part of the ranking matters.

**Metrics to monitor.**
- Model: AUC, calibration (predicted vs actual conversion by score decile), precision@k at real capacity.
- Data: feature drift (source mix, pages_viewed distribution), share of missing values.
- Business: conversion by score band, speed-to-contact for the top band, rep override rate.
- Feedback-loop risk: leads never called never convert, so keep a randomized holdout or the model learns its own routing.

## Roadmap

- Precision@k and lift report command
- Calibration plot and isotonic calibration
- Time-based train/test split to mimic production drift
- CSV ingest with schema validation

## License

MIT
