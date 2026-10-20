# AGENTS.md — MenuDemand Forecast

This file defines the operating rules for coding agents, AI assistants, contributors, and automated edits in this repository.

Its purpose is to prevent future changes from silently altering the project claim, champion model, data semantics, evaluation protocol, or historical metrics.

---

## 1. Project Identity

**Project:** MenuDemand Forecast  
**Portfolio milestone:** 13%  
**Repository:** `MENSTRUE/MenuDemand-Forecast`  
**Primary task:** next-day product-level unit demand forecasting  
**Forecast horizon:** one day ahead  
**Granularity:** Product × Day  
**Operational champion:** Rolling-28  
**Best learned model:** LightGBM L1  
**Selection metric:** validation MAE  
**Official test protocol:** rolling one-day-ahead  
**Application:** Streamlit

The project is an MVP and must be represented as such.

---

## 2. Source-of-Truth Hierarchy

When files disagree, use this priority order.

### Tier 1 — machine-readable experiment artifacts

1. `reports/final_summary.json`
2. `reports/model_selection.json`
3. `reports/official_test_metrics.json`
4. `reports/forecast_config.json`
5. `reports/raw_audit.json`
6. `reports/item_parsing_audit.json`
7. `reports/product_day_panel_audit.json`
8. `reports/granularity_comparison.json`
9. `reports/validation_model_comparison.csv`
10. `reports/official_test_model_comparison.csv`

Do not hand-edit historical numbers to make results look better.

### Tier 2 — experiment notebook

`notebooks/menudemand-forecast.ipynb`

### Tier 3 — application implementation

- `src/inference.py`
- `src/utils.py`
- `app/app.py`
- `tests/test_inference.py`

### Tier 4 — documentation

- `README.md`
- comments
- app copy
- portfolio captions

Documentation must reflect artifacts, not redefine them.

---

## 3. Locked Historical Facts

Do not change these facts without rerunning the experiment and regenerating reports.

### Dataset audit

- transaction rows: **53,820**
- catalog rows: **96**
- parsed units: **61,973**
- active products: **94**
- outlets: **3**
- calendar days: **272**
- date range: **2025-01-02 → 2025-09-30**
- missing calendar days: **0**
- duplicate receipts: **0**
- unmapped items: **0**
- loyalty receipts: **1,733**
- inactive catalog: `Add on Aren`, `Add on Simple Syrup`

### Product × Day panel

- rows: **25,568**
- zero-demand rate: **0.0962531289**
- mean demand: **2.4238501252**
- median: **2**
- p90: **5**
- p95: **5**
- p99: **7**
- max: **10**

### Granularity audit

Product × Day:
- zero rate: **0.0962531289**
- mean: **2.4238501252**

Outlet × Product × Day:
- zero rate: **0.4682285148**
- mean: **0.8079500417**

### Temporal split

Train: **17,202 rows**, 2025-01-30 → 2025-07-31  
Validation: **2,914 rows**, 2025-08-01 → 2025-08-31  
Test: **2,820 rows**, 2025-09-01 → 2025-09-30

---

## 4. Champion Contract

### Operational champion — Rolling-28

Validation:
- MAE: **1.2894769095**
- RMSE: **1.5826754985**
- WAPE: **0.5479853747**
- Bias: **+0.2498406707**

### Best learned model — LGBM-L1

Validation:
- MAE: **1.3267588821**
- RMSE: **1.6206689007**
- WAPE: **0.5638289897**
- Bias: **+0.4636881168**

### Selection rule

Champion selection uses **validation MAE only**.

Do not promote LightGBM merely because it has slightly lower held-out test MAE.

---

## 5. Official Test Contract

Held-out period: **2025-09-01 → 2025-09-30**  
Protocol: **rolling one-day-ahead**

### Rolling-28

- MAE: **1.2314842958**
- RMSE: **1.5406887343**
- WAPE: **0.5460354897**
- Bias: **+0.0414640324**
- MeanActual: **2.2553191489**
- MeanPred: **2.2967831814**
- N: **2,820**

### LGBM-L1

- MAE: **1.2234185547**
- RMSE: **1.5437866936**
- WAPE: **0.5424591705**
- Bias: **-0.0273251368**
- MeanActual: **2.2553191489**
- MeanPred: **2.2279940121**
- N: **2,820**

Historical test results must not be silently recomputed and presented as the same experiment.

---

## 6. Claim Boundaries

### Allowed

- next-day menu-item demand forecasting
- product-level daily unit forecasting
- Product × Day forecasting
- strict temporal held-out evaluation
- rolling one-day-ahead evaluation
- baseline-versus-LightGBM comparison
- lag/rolling/calendar features
- MVP
- Streamlit inference app

### Forbidden unless newly validated

- inventory optimization
- optimal ordering
- stockout prediction
- causal promo uplift
- causal price elasticity
- universal restaurant forecasting
- long-horizon accuracy
- outlet-level accuracy
- guaranteed future sales
- guaranteed savings
- “accuracy” percentages derived from MAE/WAPE
- claims that the test set selected the champion

---

## 7. Target Semantics

Primary target: **daily units sold per product**.

Do not silently switch to revenue, receipt count, profit, inventory, stock level, or customer count.

---

## 8. Data Storage Rules

Raw source data is not currently redistributed.

Do not add third-party raw data unless redistribution rights and privacy requirements are verified.

Directory ownership:

- reports → `reports/`
- visualizations → `assets/`
- models → `models/`
- notebook → `notebooks/`
- ephemeral outputs → `tests/results/`

Do not commit user-uploaded sales CSVs.

---

## 9. Input Schema Contract

Required:

- `Date`
- `ProductId`
- `ProductDisplay`
- `Category`
- `Price`
- `demand`

At least **28 distinct dates per product**.

Exactly one row per `ProductId + Date`.

Reject duplicates; do not silently drop them.

Demand must be numeric and non-negative.

Forecast date = `max(Date) + 1 calendar day`.

---

## 10. Operational Inference Rule

Default:

```text
forecast next day = mean(previous 28 daily demand values)
```

Implementation: `src/inference.py::forecast_rolling_28`

The Rolling-28 path must not require the LightGBM artifact.

`forecast_units` is continuous. `forecast_units_rounded` is display-only.

---

## 11. Learned Inference Rule

Artifact: `models/menudemand_best_learned_lgbm.joblib`

Expected feature order:

- ProductId
- Category
- Price
- lag_1
- lag_2
- lag_3
- lag_7
- lag_14
- lag_28
- rolling_mean_3
- rolling_mean_7
- rolling_mean_14
- rolling_mean_28
- rolling_std_7
- rolling_std_28
- expanding_mean
- day_of_week
- day_of_month
- week_of_year
- month
- is_weekend
- dow_sin
- dow_cos
- month_sin
- month_cos

Categorical: `ProductId`, `Category`.

Do not rename/reorder/change feature semantics without retraining and revalidating.

---

## 12. Leakage Policy

Every next-day feature must be available at forecast time.

Allowed:
- historical demand
- known calendar fields
- static metadata
- price only if genuinely known before forecast

Forbidden:
- next-day actual demand
- target-containing rolling windows
- future realized sales
- post-day stockout state
- future signals not genuinely known at forecast time

When uncertain, treat a feature as leakage risk.

---

## 13. Experiment History

Baselines:
- Naive-1
- Seasonal-7
- Rolling-7
- Rolling-28

Learned:
- LGBM-L1
- LGBM-Poisson
- LGBM-Tweedie

Validation MAE ranking:

1. Rolling-28 — 1.2895
2. Rolling-7 — 1.3117
3. LGBM-L1 — 1.3268
4. LGBM-Poisson — 1.4016
5. LGBM-Tweedie — 1.4258
6. Naive-1 — 1.6850
7. Seasonal-7 — 1.7152

Preserve this historical record unless a versioned successor replaces it.

---

## 14. Model Promotion Checklist

A new operational champion requires:

- [ ] declared task/horizon
- [ ] documented data period
- [ ] leakage audit
- [ ] predeclared temporal split
- [ ] Rolling-28 retained as baseline
- [ ] declared selection metric
- [ ] validation win under that rule
- [ ] test opened only after selection
- [ ] failure analysis
- [ ] regenerated reports
- [ ] inference parity test
- [ ] README update
- [ ] AGENTS update if semantics changed
- [ ] old results remain traceable

A test-only improvement is insufficient.

---

## 15. Directory Contracts

### `app/`
UI orchestration only. Reusable forecast logic belongs in `src/`.

### `src/`
Reusable inference logic.

### `models/`
Serialized models only.

### `notebooks/`
Experiment/research source. No runtime dependence on notebook state.

### `reports/`
Machine-readable experiment outputs.

### `assets/dataset/`
Dataset figures.

### `assets/training/`
Validation/training figures.

### `assets/evaluation/`
Held-out test figures.

### `assets/predictions/`
Forecast figures.

### `assets/screenshots/`
App screenshots.

### `tests/`
Automated tests.

### `tests/results/`
Ephemeral outputs; ignored except `.gitkeep`.

---

## 16. README Asset Policy

Approved current images:

- `assets/dataset/daily_total_units.png`
- `assets/dataset/top20_products_total_units.png`
- `assets/dataset/weekday_mean_units.png`
- `assets/training/validation_model_comparison_mae.png`
- `assets/training/best_learned_feature_importance_top20.png`
- `assets/evaluation/operational_test_mae_by_category.png`
- `assets/evaluation/test_actual_vs_operational_prediction.png`
- `assets/evaluation/test_daily_total_comparison.png`
- `assets/evaluation/test_operational_vs_learned_mae.png`
- `assets/predictions/next_day_top20_forecast.png`

Do not reference nonexistent screenshots.

---

## 17. Training Policy

If training changes:

1. preserve old experiment outputs;
2. create versioned new outputs;
3. rerun audit;
4. rerun baselines;
5. lock validation selection;
6. evaluate test after selection;
7. export exact metrics;
8. export failure cases;
9. export plots;
10. replace model only after promotion.

Never overwrite the model with an unvalidated candidate while leaving docs unchanged.

---

## 18. Evaluation Policy

Required regression metrics:

- MAE
- RMSE
- WAPE
- Bias

Useful supporting outputs:

- mean actual
- mean prediction
- N
- per-category MAE
- per-product MAE
- worst absolute errors

Do not use classification-style “accuracy” for this regression task.

Do not convert `1 - WAPE` into “accuracy” without a separately justified definition.

---

## 19. Failure Analysis Policy

Do not publish only aggregate metrics.

Known failure pattern: **sudden demand spikes are underpredicted**.

Examples:

- Candy Latte actual 10 vs Rolling-28 ≈ 2.07
- Espresso Houseblend actual 10 vs Rolling-28 ≈ 2.39
- Americano (Ice Arabica) actual 8 vs Rolling-28 ≈ 1.82

Future work should seek valid pre-forecast signals, not just bigger models.

---

## 20. Testing Policy

Run before merging inference/app changes:

```bash
python -m pytest -q
```

Current test intent:

- validate acceptable history
- deterministic Rolling-28 output
- next-day date calculation

Add tests for validation changes. Add parity tests before modifying LightGBM preprocessing.

---

## 21. Required Future Tests

- missing required column
- non-numeric demand
- negative demand
- invalid date
- fewer than 28 dates
- duplicate ProductId + Date
- missing model artifact
- feature-order parity
- unseen ProductId
- unseen Category
- export correctness
- one-product edge case

---

## 22. App Policy

Required product flow:

1. upload
2. validate
3. choose method
4. generate
5. inspect
6. export

Rolling-28 remains default unless a newly validated champion replaces it.

LightGBM must be labeled best learned model/comparison, not operational champion.

---

## 23. UI Claim Policy

Allowed:

- next-day forecast
- forecast units
- historical data is valid
- operational champion
- best learned model

Avoid:

- recommended order quantity
- optimal stock
- guaranteed demand
- AI recommendation when output is only a forecast

---

## 24. Dependency Policy

`requirements.txt` is the dependency contract.

Avoid unnecessary packages. Check maintenance, license, deployment compatibility, and whether existing dependencies already solve the task.

---

## 25. Privacy and Security

Never commit:

- credentials
- API keys
- private customer data
- personally identifying transaction data
- uploaded user CSVs
- local environment files

Treat uploaded sales history as potentially sensitive business data.

---

## 26. Git Policy

Preferred commits:

```text
feat: add interactive next-day forecast workflow
fix: validate duplicate product-day rows
test: cover insufficient history validation
docs: add full MenuDemand project documentation
refactor: separate forecast logic from Streamlit UI
```

Do not force-push or rewrite `main` casually.

Do not commit caches, `tests/results/*`, or private datasets.

Before destructive cleanup:

```bash
git clean -fdn
```

Review dry-run output first.

---

## 27. Documentation Update Rules

Update README when these change:

- task
- target
- granularity
- champion
- selection rule
- metrics
- schema
- model file
- app behavior
- structure
- limitations

Update AGENTS.md when these change:

- source-of-truth hierarchy
- claim boundaries
- promotion policy
- directory contracts
- testing policy
- data policy
- inference semantics

---

## 28. Reproducibility Checklist

- [ ] source documented
- [ ] source files identified
- [ ] audit exported
- [ ] preprocessing deterministic
- [ ] features documented
- [ ] temporal split exported
- [ ] baselines exported
- [ ] learned comparison exported
- [ ] selection rule exported
- [ ] test metrics exported
- [ ] model versioned
- [ ] app preprocessing matches training
- [ ] tests run
- [ ] limitations updated

---

## 29. Deployment Policy

Local:

```bash
streamlit run app/app.py
```

Hosted deployment must use repository-relative paths and include a smoke test.

Production-grade deployment additionally needs access control, logging, observability, data retention rules, artifact/version metadata, schema versioning, and rollback strategy.

---

## 30. Roadmap Guardrails

### Safe improvements

- better validation
- sample-input generation
- uncertainty intervals
- category filtering
- multi-window backtesting
- promo/event/weather/stock features
- outlet modeling
- monitoring

### Requires reframing/revalidation

- multi-day recursive forecasting
- inventory optimization
- causal promotion uplift
- outlet-level deployment
- automated purchasing
- cross-business generalization
- unseen-product cold start

Do not attach these claims to existing metrics.

---

## 31. Before → Now → Next Rule

### Before
Receipt-level history without a validated forecasting pipeline.

### Now
Audited Product × Day next-day forecasting MVP with Rolling-28 champion, LightGBM comparison, temporal evaluation, Streamlit inference, and explicit limits.

### Next
Improve information quality and backtesting before expanding deployment claims.

---

## 32. Definition of Done — Code Change

A code change is done when:

- behavior works
- paths are repository-relative
- tests pass
- new behavior is tested where practical
- app starts
- claim boundaries remain correct
- historical metrics are not silently changed
- docs are updated when behavior changes
- no secrets/local artifacts are committed

---

## 33. Definition of Done — Model Change

A model change is done only when:

- data audit complete
- leakage reviewed
- baselines included
- validation rule predeclared
- candidate wins by declared rule
- test evaluated after selection
- failure analysis exported
- reports regenerated
- inference parity confirmed
- artifact saved
- docs updated
- old results traceable

---

## 34. Definition of Done — Documentation Change

Documentation is done when:

- numbers match report artifacts
- champion is correct
- test is not described as selection
- limitations are visible
- unsupported claims are absent
- asset links resolve
- repo structure matches reality
- commands match current files
- dataset licensing is not invented

---

## 35. Agent Safety Rule

When uncertain, preserve the current validated claim rather than making the project sound more impressive.

Optimize for correctness, reproducibility, traceability, portfolio credibility, and engineering clarity.

---

## 36. Final Non-Negotiable Principle

> **The project chooses the operational champion on validation performance, not on the held-out test.**

Current validated state:

**Rolling-28 = operational champion**

**LightGBM L1 = best learned model**

Any agent that changes this must first produce a new, versioned, leakage-audited, validation-selected experiment that justifies the change.
