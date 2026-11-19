"""Read and validate the eleven campaign extracts without modifying raw files."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ACTIONS

SCHEMA = {
    "campaigns": "campaign_id send_ts campaign_name channel outcome_window_days notes",
    "customers": "customer_id signup_date region acquisition_channel customer_tier is_member preferred_category shipping_zone marketing_consent",
    "historical_assignments": "assignment_id campaign_id customer_id send_ts assignment_source offer_type email_delivered",
    "web_sessions": "session_id customer_id session_ts device_type pages_viewed added_to_cart category_viewed traffic_source",
    "email_events": "email_event_id assignment_id customer_id campaign_id event_ts event_type",
    "orders": "order_id customer_id order_ts campaign_id order_source order_status offer_redeemed offer_type_redeemed merchandise_gross_revenue discount_amount shipping_revenue net_revenue cogs shipping_cost",
    "order_items": "order_id line_number product_id quantity unit_list_price unit_cost",
    "products": "product_id category list_price unit_cost return_rate_prior item_weight_lb",
    "returns": "return_id order_id return_ts refund_amount return_reason",
    "offer_catalog": "offer_type customer_message promotion_cost_rule",
    "policy_scoring_population": "customer_id policy_as_of_ts campaign_id",
}
TIMESTAMPS = {"campaigns": "send_ts", "customers": "signup_date",
    "historical_assignments": "send_ts", "web_sessions": "session_ts",
    "email_events": "event_ts", "orders": "order_ts", "returns": "return_ts",
    "policy_scoring_population": "policy_as_of_ts"}
KEYS = {"campaigns": ["campaign_id"], "customers": ["customer_id"],
    "historical_assignments": ["assignment_id"], "web_sessions": ["session_id"],
    "email_events": ["email_event_id"], "orders": ["order_id"],
    "order_items": ["order_id", "line_number"], "products": ["product_id"],
    "returns": ["return_id"], "offer_catalog": ["offer_type"],
    "policy_scoring_population": ["customer_id"]}
NULLABLE = {("customers", "acquisition_channel"), ("customers", "preferred_category"),
    ("web_sessions", "customer_id"), ("orders", "campaign_id"),
    ("orders", "offer_type_redeemed"), ("campaigns", "notes")}
FINANCIALS = ["merchandise_gross_revenue", "discount_amount", "shipping_revenue",
              "net_revenue", "cogs", "shipping_cost"]


class DataQualityError(ValueError):
    """Invalid input with the structured checks available as ``report``."""
    def __init__(self, report):
        self.report = report
        failures = report.loc[report.status.eq("FAIL"), ["table", "test", "n_fail"]]
        super().__init__("Data quality checks failed:\n" + failures.to_string(index=False))


def load_tables(data_dir):
    """Load raw CSVs from an explicit directory; date errors fail immediately."""
    root = Path(data_dir)
    missing = [f"{name}.csv" for name in SCHEMA if not (root / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input CSVs in {root}: {', '.join(missing)}")
    tables = {name: pd.read_csv(root / f"{name}.csv") for name in SCHEMA}
    for name, column in TIMESTAMPS.items():
        if column in tables[name]:
            tables[name][column] = pd.to_datetime(tables[name][column], errors="raise", format="mixed")
    return tables


def validate_and_clean(tables, *, tolerance=.05, outcome_days=14):
    """Return independent clean tables and a PASS/WARN quality registry.

    Exact duplicate records are removed once and counted. Different records
    sharing a primary key are rejected before any join. Anonymous web sessions
    remain in the clean extract; feature construction excludes them.
    """
    if not np.isfinite(tolerance) or tolerance < 0 or outcome_days <= 0:
        raise ValueError("Financial tolerance must be nonnegative and outcome_days positive")
    checks = []

    def check(name, table, count, *, warning=False, note=""):
        count = int(count)
        checks.append(dict(test=name, table=table, n_fail=count,
                           status="PASS" if count == 0 else "WARN" if warning else "FAIL", note=note))

    def stop_on_fail():
        report = pd.DataFrame(checks)
        if report.status.eq("FAIL").any():
            raise DataQualityError(report)

    for name, columns in SCHEMA.items():
        missing = set(columns.split()) - set(tables.get(name, pd.DataFrame()).columns)
        check("required_columns", name, len(missing), note=", ".join(sorted(missing)))
    stop_on_fail()
    clean = {}
    for name in SCHEMA:
        source = tables[name]
        clean[name] = source.drop_duplicates().copy().reset_index(drop=True)
        check("exact_duplicates_removed", name, len(source) - len(clean[name]), warning=True)
        frame = clean[name]
        keys = KEYS[name]
        check("unique_nonnull_primary_key", name, frame.duplicated(keys).sum() + frame[keys].isna().any(axis=1).sum())
        required = [c for c in SCHEMA[name].split() if (name, c) not in NULLABLE]
        check("required_values_present", name, frame[required].isna().any(axis=1).sum())
        if name in TIMESTAMPS:
            column = TIMESTAMPS[name]
            parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed")
            check("valid_timestamps", name, parsed.isna().sum())
            frame[column] = parsed
    ha = clean["historical_assignments"]
    check("unique_customer_campaign_assignment", "historical_assignments", ha.duplicated(["customer_id", "campaign_id"]).sum())
    stop_on_fail()
    customers, campaigns = clean["customers"], clean["campaigns"]
    orders, items, returns = clean["orders"], clean["order_items"], clean["returns"]
    email, web, policy = clean["email_events"], clean["web_sessions"], clean["policy_scoring_population"]

    for name, column, parent, parent_key in [
        ("historical_assignments", "customer_id", "customers", "customer_id"),
        ("historical_assignments", "campaign_id", "campaigns", "campaign_id"),
        ("orders", "customer_id", "customers", "customer_id"),
        ("orders", "campaign_id", "campaigns", "campaign_id"),
        ("order_items", "order_id", "orders", "order_id"),
        ("order_items", "product_id", "products", "product_id"),
        ("returns", "order_id", "orders", "order_id"),
        ("email_events", "assignment_id", "historical_assignments", "assignment_id"),
        ("web_sessions", "customer_id", "customers", "customer_id"),
        ("policy_scoring_population", "customer_id", "customers", "customer_id"),
        ("policy_scoring_population", "campaign_id", "campaigns", "campaign_id"),
    ]:
        check(f"foreign_key:{column}", name, (~clean[name][column].dropna().isin(clean[parent][parent_key])).sum())
    numeric = {"orders": FINANCIALS + ["offer_redeemed"],
        "order_items": ["quantity", "unit_list_price", "unit_cost", "line_number"],
        "returns": ["refund_amount"], "products": ["list_price", "unit_cost", "return_rate_prior", "item_weight_lb"],
        "customers": ["is_member", "marketing_consent"],
        "historical_assignments": ["email_delivered"], "campaigns": ["outcome_window_days"],
        "web_sessions": ["pages_viewed", "added_to_cart"]}
    for name, columns in numeric.items():
        clean[name][columns] = clean[name][columns].apply(pd.to_numeric, errors="coerce")
        check("finite_numeric_values", name, (~np.isfinite(clean[name][columns])).any(axis=1).sum())
    stop_on_fail()
    for name, column in [("customers", "is_member"), ("customers", "marketing_consent"),
                         ("historical_assignments", "email_delivered"), ("orders", "offer_redeemed"),
                         ("web_sessions", "added_to_cart")]:
        check(f"binary:{column}", name, (~clean[name][column].isin([0, 1])).sum())
    check("valid_assignment_source", "historical_assignments", (~ha.assignment_source.isin(["exploration", "business_rule"])).sum())
    check("valid_event_type", "email_events", (~email.event_type.isin(["open", "click"])).sum())
    check("valid_order_status", "orders", (~orders.order_status.isin(["completed", "cancelled"])).sum())
    check("valid_order_source", "orders", (~orders.order_source.isin(["organic_preperiod", "campaign_attributed"])).sum())
    check("campaign_missing_only_for_organic", "orders", (orders.campaign_id.isna() != orders.order_source.eq("organic_preperiod")).sum())
    check("redeemed_type_consistent_with_flag", "orders", (orders.offer_type_redeemed.isna() != orders.offer_redeemed.eq(0)).sum())
    check("anonymous_sessions_excluded_from_features", "web_sessions", web.customer_id.isna().sum(), warning=True)
    check("positive_item_quantity_and_price", "order_items", ((items.quantity <= 0) | (items.unit_list_price <= 0) | (items.unit_cost < 0)).sum())
    check("nonnegative_order_financials", "orders", (orders[FINANCIALS] < 0).any(axis=1).sum())
    check("cancelled_orders_have_zero_financials", "orders", orders.loc[orders.order_status.eq("cancelled"), FINANCIALS].ne(0).any(axis=1).sum())
    check("order_net_revenue_reconciles", "orders", (abs(orders.net_revenue - orders.merchandise_gross_revenue + orders.discount_amount - orders.shipping_revenue) > tolerance).sum())
    check("discount_not_above_gross", "orders", (orders.discount_amount > orders.merchandise_gross_revenue + tolerance).sum())
    check("positive_refunds", "returns", returns.refund_amount.le(0).sum())
    totals = returns.groupby("order_id").refund_amount.sum()
    check("cumulative_refunds_not_above_net", "returns", (totals > orders.set_index("order_id").net_revenue.reindex(totals.index) + tolerance).sum())
    linked_returns = returns.merge(orders, on="order_id", validate="many_to_one")
    check("returns_after_order", "returns", linked_returns.return_ts.le(linked_returns.order_ts).sum())
    check("returns_on_completed_orders", "returns", linked_returns.order_status.ne("completed").sum())
    product_totals = items.assign(item_gross=items.quantity * items.unit_list_price,
                                 item_cost=items.quantity * items.unit_cost).groupby("order_id")[["item_gross", "item_cost"]].sum()
    reconciliation = orders.loc[orders.order_status.eq("completed")].join(product_totals, on="order_id")
    check("completed_order_has_items", "orders", reconciliation.item_gross.isna().sum())
    check("item_gross_reconciles", "order_items", (abs(reconciliation.merchandise_gross_revenue - reconciliation.item_gross) > tolerance).sum())
    check("item_cost_reconciles", "order_items", (abs(reconciliation.cogs - reconciliation.item_cost) > tolerance).sum())
    catalog = set(clean["offer_catalog"].offer_type)
    check("four_official_offers", "offer_catalog", len(catalog ^ set(ACTIONS)))
    check("valid_assigned_offer", "historical_assignments", (~ha.offer_type.isin(ACTIONS)).sum())
    check("valid_redeemed_offer", "orders", (~orders.offer_type_redeemed.dropna().isin(set(ACTIONS) - {"NO_OFFER"})).sum())
    expected_rate = orders.offer_type_redeemed.map({"PCT_10": .1, "PCT_20": .2, "FREE_SHIP": 0}).fillna(0)
    actual_rate = orders.discount_amount / orders.merchandise_gross_revenue.where(orders.merchandise_gross_revenue > 0)
    check("discount_matches_redeemed_offer", "orders", (abs(actual_rate - expected_rate) > .005).sum())
    check("free_ship_has_zero_customer_shipping_fee", "orders", orders.loc[orders.offer_type_redeemed.eq("FREE_SHIP"), "shipping_revenue"].gt(0).sum())
    assignment_campaign = ha.merge(campaigns[["campaign_id", "send_ts"]], on="campaign_id", suffixes=("", "_campaign"), validate="many_to_one")
    check("assignment_send_matches_campaign", "historical_assignments", assignment_campaign.send_ts.ne(assignment_campaign.send_ts_campaign).sum())
    check("fixed_outcome_horizon", "campaigns", campaigns.outcome_window_days.ne(outcome_days).sum())
    signup = ha.merge(customers[["customer_id", "signup_date"]], on="customer_id", validate="many_to_one")
    check("signup_before_campaign", "historical_assignments", signup.signup_date.gt(signup.send_ts).sum())
    policy_time = policy.merge(campaigns[["campaign_id", "send_ts"]], on="campaign_id", validate="many_to_one")
    check("policy_cutoff_not_after_send", "policy_scoring_population", policy_time.policy_as_of_ts.gt(policy_time.send_ts).sum())
    policy_signup = policy.merge(customers[["customer_id", "signup_date"]], on="customer_id", validate="many_to_one")
    check("signup_before_policy_cutoff", "policy_scoring_population", policy_signup.signup_date.gt(policy_signup.policy_as_of_ts).sum())
    linked = orders.loc[orders.campaign_id.notna()].merge(ha, on=["customer_id", "campaign_id"], how="left", validate="many_to_one")
    check("campaign_order_has_assignment", "orders", linked.assignment_id.isna().sum())
    check("campaign_order_within_half_open_window", "orders", (~(linked.order_ts.ge(linked.send_ts) & linked.order_ts.lt(linked.send_ts + pd.Timedelta(days=outcome_days)))).sum())
    check("redemption_matches_assignment", "orders", (linked.offer_type_redeemed.notna() & linked.offer_type_redeemed.ne(linked.offer_type)).sum())
    check("organic_orders_precede_first_campaign", "orders", orders.loc[orders.order_source.eq("organic_preperiod"), "order_ts"].ge(campaigns.send_ts.min()).sum())
    # Windows are disjoint in this project. Reject overlap rather than silently
    # crediting the same behavior to multiple concurrent campaign exposures.
    sends = campaigns.loc[campaigns.campaign_id.isin(ha.campaign_id), "send_ts"].sort_values()
    check("nonoverlapping_historical_windows", "campaigns", sends.shift(-1).lt(sends + pd.Timedelta(days=outcome_days)).sum())
    linked_email = email.merge(ha, on="assignment_id", suffixes=("", "_assignment"), validate="many_to_one")
    check("email_identity_matches_assignment", "email_events", (linked_email.customer_id.ne(linked_email.customer_id_assignment) | linked_email.campaign_id.ne(linked_email.campaign_id_assignment)).sum())
    check("email_events_after_delivered_send", "email_events", (linked_email.event_ts.lt(linked_email.send_ts) | linked_email.email_delivered.ne(1)).sum())
    stop_on_fail()
    clean["orders"] = orders.assign(is_completed=orders.order_status.eq("completed"))
    clean["web_sessions"] = web.assign(customer_id=web.customer_id.astype("Int64"), is_identified=web.customer_id.notna())
    return clean, pd.DataFrame(checks)
