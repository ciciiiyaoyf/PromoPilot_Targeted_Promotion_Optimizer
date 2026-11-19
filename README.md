# PromoPilot: Targeted Promotion Optimizer

A standalone Python pipeline for deciding which customers should receive no
offer, 10% off, 20% off, or free shipping. It forecasts 14-day net contribution
margin (NCM), then maximizes **predicted incremental NCM** within the campaign's
budget and contact limits.

The purchase-propensity benchmark is one logistic regression. The promotion
policy uses an offer-conditioned, two-part XGBoost model and a separate MILP
allocator. Model predictions, historical policy estimates, and actual campaign
results are different quantities; this project does not contain realized July
campaign outcomes.

## Run locally

Use Python 3.13, the version used for verification. From this repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest
python -m promopilot validate --data-dir "C:/path/to/data/raw"
python -m promopilot run --data-dir "C:/path/to/data/raw" --output-dir artifacts
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.
Runtime dependencies are pinned in [pyproject.toml](pyproject.toml). The installed
`promopilot` command accepts the same arguments as `python -m promopilot`.

The original project extracts must be supplied locally. They are **not included
in this public repository**, and there is no automatic data download. Place
these eleven CSVs together in the directory passed to `--data-dir`:

```text
campaigns.csv                 customers.csv
historical_assignments.csv    web_sessions.csv
email_events.csv              orders.csv
order_items.csv               products.csv
returns.csv                   offer_catalog.csv
policy_scoring_population.csv
```

The required columns and validation rules are defined in
[data.py](promopilot/data.py). Raw data, fitted models, and generated results
are excluded from version control. Tests use small synthetic inputs and do not
require the project extracts.

## Read the project

Start with [METHODS.md](METHODS.md) for the business objective, time boundaries,
data handling, model interpretation, evaluation, and measurement design.

| Module | Responsibility |
| --- | --- |
| `data.py`, `outcomes.py` | Validate the extracts and construct reconciled 14-day labels. |
| `features.py`, `coverage.py` | Build decision-time features and disclose transaction coverage. |
| `models.py` | Fit the logistic baseline, signed-profit model, and promotion-cost models. |
| `allocation.py` | Reserve measurement controls and allocate offers under all constraints. |
| `evaluation.py` | Measure prediction quality and estimate historical policy value. |
| `pipeline.py`, `cli.py` | Run the workflow and write its audit outputs. |

The main outputs are:

- `policy_submission.csv`: one recommended action per scoring customer.
- `measurement_assignments.csv`: randomized control/policy arms, kept separate
  from customers the policy itself assigns no offer.
- `action_values.csv`: all four predicted action values and promotion costs.
- `data_quality.csv`, `transaction_coverage.csv`: validation and coverage evidence.
- `model_metrics.csv`, `policy_evaluation.csv`: predictive and policy evaluation.
- `run_summary.json`, `report.md`: configuration and readable results.

See `report.md` for the results of each run and [METHODS.md](METHODS.md) for
their interpretation.

## Provenance

PromoPilot builds on the
[Walmart Customer Analysis Capstone](https://github.com/ciciiiyaoyf/Walmart-Customer-Analysis-Capstone)
and the underlying [team project](https://github.com/Xingyao0508/Capstone).
