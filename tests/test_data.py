import numpy as np
import pandas as pd
import pytest

from promopilot.data import DataQualityError, load_tables, validate_and_clean
from promopilot.outcomes import build_outcomes, inferred_waived_shipping_fee


def test_exact_duplicates_removed_without_mutating_raw_or_double_counting(raw_tables):
    raw_tables["order_items"] = pd.concat([raw_tables["order_items"], raw_tables["order_items"].iloc[[0]]], ignore_index=True)
    clean, report = validate_and_clean(raw_tables)
    assert len(raw_tables["order_items"]) == 4
    assert len(clean["order_items"]) == 3
    assert not report.status.eq("FAIL").any()
    assert report.query("table == 'order_items' and test == 'exact_duplicates_removed'").n_fail.item() == 1
    assert build_outcomes(clean).net_revenue_14d.sum() == 140


def test_conflicting_duplicate_keys_fail_before_joins(raw_tables):
    conflict = raw_tables["order_items"].iloc[[0]].assign(quantity=3)
    raw_tables["order_items"] = pd.concat([raw_tables["order_items"], conflict], ignore_index=True)
    with pytest.raises(DataQualityError, match="unique_nonnull_primary_key"):
        validate_and_clean(raw_tables)


def test_foreign_key_violation_is_rejected(raw_tables):
    raw_tables["order_items"].loc[0, "product_id"] = 999
    with pytest.raises(DataQualityError, match="foreign_key:product_id"):
        validate_and_clean(raw_tables)


def test_nonfinite_financial_values_are_rejected(raw_tables):
    raw_tables["orders"].loc[0, "net_revenue"] = np.inf
    with pytest.raises(DataQualityError, match="finite_numeric_values"):
        validate_and_clean(raw_tables)


def test_cancelled_order_cannot_carry_revenue(raw_tables):
    raw_tables["orders"].loc[3, ["net_revenue", "merchandise_gross_revenue"]] = 10
    with pytest.raises(DataQualityError, match="cancelled_orders_have_zero_financials"):
        validate_and_clean(raw_tables)


def test_item_reconciliation_rejects_wrong_order_total(raw_tables):
    raw_tables["order_items"].loc[0, "unit_list_price"] = 51.
    with pytest.raises(DataQualityError, match="item_gross_reconciles"):
        validate_and_clean(raw_tables)


def test_cumulative_refunds_cannot_exceed_order_revenue(raw_tables):
    raw_tables["returns"].loc[:, "refund_amount"] = 50.
    with pytest.raises(DataQualityError, match="cumulative_refunds_not_above_net"):
        validate_and_clean(raw_tables)


def test_return_before_purchase_and_email_identity_mismatch_are_rejected(raw_tables):
    raw_tables["returns"].loc[0, "return_ts"] = pd.Timestamp("2025-01-02")
    raw_tables["email_events"].loc[0, "customer_id"] = 2
    with pytest.raises(DataQualityError) as error:
        validate_and_clean(raw_tables)
    failures = set(error.value.report.loc[error.value.report.status.eq("FAIL"), "test"])
    assert {"returns_after_order", "email_identity_matches_assignment"} <= failures


def test_repeated_customer_exposures_remain_separate_and_late_refunds_are_disclosed(raw_tables):
    clean, _ = validate_and_clean(raw_tables)
    out = build_outcomes(clean).set_index("assignment_id")
    assert out.order_count_14d.to_dict() == {11: 1, 12: 1}
    assert out.ncm_14d.to_dict() == {11: 16., 12: 16.}
    assert out.loc[11, "ncm_14d_all_returns"] == 11.
    assert out.loc[11, "post_window_refund_amount"] == 5.
    assert out.promo_cost_14d.to_dict() == {11: 10., 12: 5.99}
    assert clean["web_sessions"].is_identified.tolist() == [True, False]


def test_outcome_end_is_exclusive_and_asof_requires_mature_windows(raw_tables):
    clean, _ = validate_and_clean(raw_tables)
    # Exercise outcome boundaries independently of the validation rejection of
    # campaign-attributed orders outside their supplied assignment windows.
    clean["orders"].loc[0, "order_ts"] = pd.Timestamp("2025-01-15 10:00")
    out = build_outcomes(clean).set_index("assignment_id")
    assert out.loc[11, "purchase_14d"] == 0
    assert out.loc[11, "ncm_14d"] == 0
    with pytest.raises(ValueError, match="mature"):
        build_outcomes(clean, as_of_ts="2025-02-10")


def test_shipping_cutoff_is_inclusive_and_financial_parameters_checked():
    np.testing.assert_allclose(inferred_waived_shipping_fee([74.99, 75., 75.01]), [5.99, 0., 0.])
    with pytest.raises(ValueError, match="nonnegative"):
        inferred_waived_shipping_fee([50], standard_fee=-1)


def test_missing_input_files_and_invalid_dates_fail_early(tmp_path, raw_tables):
    with pytest.raises(FileNotFoundError, match="campaigns.csv"):
        load_tables(tmp_path)
    raw_tables["campaigns"]["send_ts"] = raw_tables["campaigns"].send_ts.astype(object)
    raw_tables["campaigns"].loc[0, "send_ts"] = "bad-date"
    with pytest.raises(DataQualityError, match="valid_timestamps"):
        validate_and_clean(raw_tables)
