"""Time-split and CSV-to-deliverable integration, without repeating model fits."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from promopilot import pipeline
from promopilot.cli import main
from promopilot.config import ACTIONS, PipelineConfig


def _write_extracts(tables, directory):
    directory.mkdir()
    for name, frame in tables.items():
        frame.to_csv(directory / f"{name}.csv", index=False)


def test_training_before_uses_mature_labels_and_preserves_decision_features():
    cutoff = pd.Timestamp("2025-02-01")
    history = pd.DataFrame({
        "assignment_id": [11, 12, 13, 14],
        "customer_id": [1, 1, 2, 2],
        "send_ts": pd.to_datetime(["2025-01-01", "2025-01-18", "2025-01-20", "2025-02-01"]),
        "outcome_end_ts": pd.to_datetime(["2025-01-15", "2025-02-01", "2025-02-03", "2025-02-15"]),
        "observed_orders_30d": [0, 2, 7, 9],
        "ncm_14d": [1., -2., 100., 200.],
    })
    original = history.copy(deep=True)
    mature = pipeline.training_before(history, cutoff)
    pd.testing.assert_frame_equal(mature, original.iloc[:2])
    mature.loc[11 == mature.assignment_id, "ncm_14d"] = 999.
    pd.testing.assert_frame_equal(history, original)
    with pytest.raises(ValueError, match="No mature training outcomes"):
        pipeline.training_before(history, "2025-01-10")


@pytest.mark.parametrize("development", ["CP06", "CP07", "CP08"])
def test_threshold_selection_cannot_use_unavailable_or_evaluation_outcomes(development, monkeypatch):
    history = pd.DataFrame({
        "campaign_id": ["CP06", "CP07", "CP08"],
        "send_ts": pd.to_datetime(["2025-01-20", "2025-02-01", "2025-03-01"]),
        "outcome_end_ts": pd.to_datetime(["2025-02-03", "2025-02-15", "2025-03-15"]),
    })

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Development labels must be screened before fitting")

    monkeypatch.setattr(pipeline, "fit_logistic_baseline", unexpected_fit)
    config = PipelineConfig(development_campaigns=(development,), evaluation_campaign="CP07")
    with pytest.raises(ValueError, match="Development outcomes must mature before"):
        pipeline._development_thresholds(history, config)


def test_run_rejects_raw_output_collision_before_writing(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    sentinel = source / "orders.csv"
    sentinel.write_text("keep original extract", encoding="utf-8")
    with pytest.raises(ValueError, match="must differ"):
        pipeline.run(source, source / ".")
    assert sentinel.read_text(encoding="utf-8") == "keep original extract"
    assert list(source.iterdir()) == [sentinel]


def test_run_rejects_unmatured_scoring_labels_before_modeling(raw_tables, tmp_path, monkeypatch):
    # CP02 labels do not finish until February 15; this decision is February 10.
    raw_tables["policy_scoring_population"]["policy_as_of_ts"] = pd.Timestamp("2025-02-10 10:00")
    source = tmp_path / "raw"
    _write_extracts(raw_tables, source)

    def unexpected_features(*args, **kwargs):
        pytest.fail("Incomplete outcome windows must fail before features or model fitting")

    monkeypatch.setattr(pipeline, "build_features", unexpected_features)
    with pytest.raises(ValueError, match="mature before the scoring cutoff"):
        pipeline.run(source, tmp_path / "out")
    assert not (tmp_path / "out" / "policy_submission.csv").exists()


def test_validate_cli_reports_invalid_extract_with_failure_exit(raw_tables, tmp_path, capsys):
    raw_tables["order_items"].loc[0, "quantity"] = -1
    source = tmp_path / "raw"
    _write_extracts(raw_tables, source)
    with pytest.raises(SystemExit) as failure:
        main(["validate", "--data-dir", str(source)])
    assert failure.value.code == 2
    assert "positive_item_quantity_and_price" in capsys.readouterr().err


def test_csv_to_allocation_preserves_population_context_and_serializes_audits(raw_tables, tmp_path, monkeypatch):
    """Real validation/features/outcomes/coverage/MILP/files; only fitting is stubbed."""
    raw_tables["policy_scoring_population"] = pd.DataFrame({
        "customer_id": [2, 1],
        "policy_as_of_ts": pd.to_datetime(["2025-02-20 10:00"] * 2),
        "campaign_id": ["CP08_POLICY"] * 2,
    })
    source, destination = tmp_path / "raw", tmp_path / "output"
    _write_extracts(raw_tables, source)
    fingerprints = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
    captured = {}

    def development(history, config):
        captured["history"] = history.copy(deep=True)
        return {"logistic": .5, "hurdle_purchase": .5}, []

    def evaluation(history, scoring_size, thresholds, config):
        assert scoring_size == 2
        assert thresholds == {"logistic": .5, "hurdle_purchase": .5}
        return pd.DataFrame([
            {"model": "logistic", "scope": "all_assignments", "average_precision": .5, "recall": .5},
            {"model": "hurdle_xgb_half", "scope": "pooled", "rmse": 1., "mae": 1., "r2": 0.},
        ]), pd.DataFrame({"policy": ["ALL_NO_OFFER"]})

    class PurchaseModel:
        def predict_proba(self, features):
            assert list(features) == pipeline.NUMERIC + pipeline.CATEGORICAL
            return np.tile([.5, .5], (len(features), 1))

    def fit_logistic(history, numeric, categorical):
        captured["fit"] = history.copy(deep=True)
        return PurchaseModel()

    def fit_profit(history, numeric, categorical):
        assert history.assignment_source.eq("exploration").all()
        return "profit-models"

    def fit_cost(history, numeric, categorical, selection_campaign):
        assert history.assignment_source.eq("exploration").all()
        return {action: SimpleNamespace(selection_={"model": "test_constant"}) for action in ACTIONS[1:]}

    monkeypatch.setattr(pipeline, "_development_thresholds", development)
    monkeypatch.setattr(pipeline, "_evaluate", evaluation)
    monkeypatch.setattr(pipeline, "fit_logistic_baseline", fit_logistic)
    monkeypatch.setattr(pipeline, "fit_profit_models", fit_profit)
    monkeypatch.setattr(pipeline, "fit_cost_models", fit_cost)
    monkeypatch.setattr(pipeline, "predict_actions", lambda models, features: np.tile([10., 15., 20., 25.], (len(features), 1)))
    monkeypatch.setattr(pipeline, "predict_costs", lambda models, features: np.tile([0., 3., 4., 5.], (len(features), 1)))
    summary = pipeline.run(source, destination, PipelineConfig(bootstrap_draws=2))

    historical = captured["history"].set_index("assignment_id")
    assert historical.loc[11, "observed_prior_order_count"] == 1
    assert historical.loc[12, "observed_prior_order_count"] == 2
    assert historical.loc[11, "ncm_14d"] == pytest.approx(16.)
    assert historical.loc[12, "ncm_14d"] == pytest.approx(16.)
    pd.testing.assert_frame_equal(captured["fit"], captured["history"])
    assert summary["historical_decisions"] == 2
    assert summary["scoring_customers"] == 2
    assert summary["projected_incremental_ncm"] == pytest.approx(15.)
    assert summary["allocation_counts"]["FREE_SHIP"] == 1
    assert summary["allocation_counts"]["NO_OFFER"] == 1
    submission = pd.read_csv(destination / "policy_submission.csv")
    assert submission.columns.tolist() == ["customer_id", "recommended_action", "estimated_incremental_ncm", "estimated_promo_cost"]
    assert set(submission.customer_id) == {1, 2}
    values = pd.read_csv(destination / "action_values.csv")
    assert len(values) == 8 and values.campaign_id.eq("CP08_POLICY").all()
    assert pd.to_datetime(values.policy_as_of_ts).eq(pd.Timestamp("2025-02-20 10:00")).all()
    measurement = pd.read_csv(destination / "measurement_assignments.csv")
    assert measurement.randomized_arm.value_counts().to_dict() == {"CONTROL": 1, "POLICY": 1}
    saved = json.loads((destination / "run_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert saved["feature_cutoff_audit"]["historical"]
    assert (destination / "report.md").is_file()
    assert fingerprints == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source.iterdir()}
