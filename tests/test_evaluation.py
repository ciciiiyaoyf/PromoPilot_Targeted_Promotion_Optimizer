"""Diagnostic operating points and identification checks for logged policies."""
import numpy as np
import pandas as pd
import pytest

from promopilot.config import ACTIONS
from promopilot.evaluation import (
    classification_metrics, dr_components, evaluate_policy, regression_metrics,
    select_threshold, value_table,
)


@pytest.fixture
def logged_campaign():
    actions = np.repeat(ACTIONS, 20)
    n = len(actions)
    data = pd.DataFrame({
        "assignment_id": [f"a{i}" for i in range(n)],
        "customer_id": [f"c{i}" for i in range(n)],
        "campaign_id": "CP07", "send_ts": pd.Timestamp("2025-06-01"),
        "offer_type": actions,
        "ncm_14d": np.repeat([3., 7., 10., 5.], 20),
        "promo_cost_14d": np.repeat([0., 1., 2., 1.], 20),
    })
    # Deliberately imperfect nuisance values: DR must correct factual residuals.
    mu = np.tile([1., 2., 4., 2.], (n, 1))
    costs = np.tile([0., .5, 1., .5], (n, 1))
    return data, mu, costs


def test_frozen_threshold_confusion_and_ranking():
    y, p = [0, 1, 1, 0], [.1, .4, .8, .7]
    threshold = select_threshold(y, p)
    assert threshold == .4
    metrics = classification_metrics(y, p, threshold)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 0, 2)
    assert metrics["recall"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(.8)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    # Evaluation uses the frozen operating point; it never selects a new one.
    assert classification_metrics(y, p, .9)["recall"] == 0


def test_signed_ncm_error_denominators_and_zero_targets():
    metrics = regression_metrics([-10., 10.], [-9., 12.])
    assert metrics["mean_error"] == 1.5
    assert metrics["total_error"] == 3
    assert metrics["wape"] == pytest.approx(.15)
    assert metrics["relative_total_error"] == pytest.approx(.15)
    assert np.isnan(regression_metrics([0., 0.], [1., -1.])["wape"])


def test_dr_correction_and_identical_no_offer_policy(logged_campaign):
    data, mu, costs = logged_campaign
    selected = np.repeat("PCT_10", len(data))
    phi, weights = dr_components(data.ncm_14d, data.offer_type, selected, mu, np.full(len(data), .25))
    assert phi.mean() == 7  # Direct 2 plus randomized correction 5.
    assert weights.sum() == len(data)
    allocation = data[["assignment_id", "customer_id"]].assign(
        recommended_action="PCT_10", estimated_promo_cost=.5).iloc[::-1]
    summary, individual = evaluate_policy(data, allocation, mu, costs, "ten_percent", "CP07", draws=100)
    assert summary["dr_incremental_ncm_per_customer"] == pytest.approx(4.)
    assert summary["ci_low"] == pytest.approx(4.)
    assert summary["ci_high"] == pytest.approx(4.)
    assert summary["dr_promo_spend"] == pytest.approx(len(data))
    assert individual.assignment_id.tolist() == data.assignment_id.tolist()
    allocation["recommended_action"] = "NO_OFFER"
    allocation["estimated_promo_cost"] = 0.
    summary, _ = evaluate_policy(data, allocation, mu, costs, "no_offer", "CP07", draws=100)
    assert summary["dr_incremental_ncm_per_customer"] == 0
    assert summary["ci_low"] == summary["ci_high"] == 0


def test_policy_evaluation_rejects_missing_support_and_misalignment(logged_campaign):
    data, mu, costs = logged_campaign
    supported = data.offer_type.ne("FREE_SHIP").to_numpy()
    partial = data.loc[supported]
    allocation = partial[["customer_id"]].assign(recommended_action="NO_OFFER", estimated_promo_cost=0.)
    with pytest.raises(ValueError, match="support"):
        evaluate_policy(partial, allocation, mu[supported], costs[supported], "none", "CP07", draws=100)
    allocation = data[["assignment_id", "customer_id"]].assign(recommended_action="NO_OFFER", estimated_promo_cost=0.)
    allocation.loc[0, "assignment_id"] = "wrong_assignment"
    with pytest.raises(ValueError, match="assignment IDs"):
        evaluate_policy(data, allocation, mu, costs, "none", "CP07", draws=100)


def test_value_table_keeps_decision_cutoff_and_four_action_alignment(logged_campaign):
    data, mu, costs = logged_campaign
    data = data.rename(columns={"send_ts": "policy_as_of_ts"})
    values = value_table(data, mu, costs, np.full(len(data), .2))
    assert len(values) == 4 * len(data)
    assert values.groupby("customer_id").action.nunique().eq(4).all()
    assert values.policy_as_of_ts.eq(data.policy_as_of_ts.iloc[0]).all()
    assert values.loc[values.action.eq("NO_OFFER"), "estimated_incremental_ncm"].eq(0).all()
    assert values.loc[values.action.eq("PCT_10"), "estimated_incremental_ncm"].eq(1).all()
