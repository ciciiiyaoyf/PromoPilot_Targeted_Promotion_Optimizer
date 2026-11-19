"""Campaign-linked 14-day purchases, promotion costs, and signed net margin."""
import numpy as np
import pandas as pd


def inferred_waived_shipping_fee(gross, *, free_shipping_cutoff=75., standard_fee=5.99):
    """Customer shipping revenue forgone, not the fulfillment shipping cost."""
    values = np.asarray(gross, dtype=float)
    if not np.isfinite(free_shipping_cutoff) or free_shipping_cutoff <= 0:
        raise ValueError("Free-shipping cutoff must be finite and positive")
    if not np.isfinite(standard_fee) or standard_fee < 0:
        raise ValueError("Standard shipping fee must be finite and nonnegative")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Merchandise gross must be finite and nonnegative")
    return np.where(values < free_shipping_cutoff, standard_fee, 0.)


def build_outcomes(clean_tables, outcome_days=14, free_shipping_cutoff=75., standard_shipping_fee=5.99, as_of_ts=None):
    """Return one outcome row per assignment, using its attributed orders once.

    Purchases and applicable refunds use [send_ts, send_ts + 14 days).
    Cancelled and organic orders do not create campaign outcomes. Late refunds
    are disclosed separately and must not be used as future training labels.
    ``as_of_ts`` optionally limits the source snapshot and requires mature
    primary windows. The returned field names preserve the project's 14-day
    target definition, so other horizon values are explicitly rejected.
    """
    if outcome_days != 14:
        raise ValueError("This project's targets are defined on a 14-day horizon")
    inferred_waived_shipping_fee([], free_shipping_cutoff=free_shipping_cutoff, standard_fee=standard_shipping_fee)
    ha = clean_tables["historical_assignments"].assign(
        outcome_end_ts=lambda frame: frame.send_ts + pd.Timedelta(days=outcome_days))
    if not ha.assignment_id.is_unique or ha.duplicated(["customer_id", "campaign_id"]).any():
        raise ValueError("Outcome construction requires unique assignment and customer-campaign keys")
    if as_of_ts is not None and not ha.outcome_end_ts.le(pd.Timestamp(as_of_ts)).all():
        raise ValueError("Every primary outcome window must mature by as_of_ts")
    orders, returns = clean_tables["orders"], clean_tables["returns"]
    linked = ha.merge(orders, on=["customer_id", "campaign_id"], how="inner", validate="one_to_many")
    valid = linked.order_ts.ge(linked.send_ts) & linked.order_ts.lt(linked.outcome_end_ts) & linked.is_completed
    eligible = linked.loc[valid].copy()
    if not eligible.order_id.is_unique:
        raise ValueError("An order cannot be credited to multiple assignments")
    eligible["redeemed"] = eligible.offer_redeemed.eq(1) & eligible.offer_type_redeemed.eq(eligible.offer_type)
    eligible["waived_ship_fee"] = np.where(eligible.offer_type_redeemed.eq("FREE_SHIP"),
        inferred_waived_shipping_fee(eligible.merchandise_gross_revenue,
            free_shipping_cutoff=free_shipping_cutoff, standard_fee=standard_shipping_fee), 0.)
    eligible["promo_cost"] = np.where(eligible.redeemed, eligible.discount_amount + eligible.waived_ship_fee, 0.)
    groups = eligible.groupby("assignment_id")
    amounts = {f"{column}_14d": groups[column].sum() for column in
               ["discount_amount", "shipping_revenue", "net_revenue", "cogs", "shipping_cost"]}
    amounts.update(order_count_14d=groups.size(), gross_revenue_14d=groups.merchandise_gross_revenue.sum(),
                   offer_redeemed_14d=groups.redeemed.max().astype(int), promo_cost_14d=groups.promo_cost.sum())
    out = ha.set_index("assignment_id")[["offer_type", "outcome_end_ts"]].join(pd.DataFrame(amounts))
    refunds = eligible[["assignment_id", "order_id", "outcome_end_ts"]].merge(returns, on="order_id", validate="one_to_many")
    if as_of_ts is not None:
        refunds = refunds.loc[refunds.return_ts.le(pd.Timestamp(as_of_ts))]
    in_window = refunds.return_ts.lt(refunds.outcome_end_ts)
    for prefix, subset in [("in_window", refunds.loc[in_window]), ("post_window", refunds.loc[~in_window])]:
        grouped = subset.groupby("assignment_id").refund_amount
        out[f"{prefix}_refund_amount"] = grouped.sum()
        out[f"{prefix}_return_count"] = grouped.size()
    numeric = out.columns.difference(["offer_type", "outcome_end_ts"])
    out[numeric] = out[numeric].fillna(0)
    out["refund_amount_applicable"] = out.in_window_refund_amount
    out["total_linked_refund_amount"] = out.in_window_refund_amount + out.post_window_refund_amount
    out["promo_cost_inferred"] = (out.offer_type.eq("FREE_SHIP") & out.offer_redeemed_14d.eq(1)).astype(int)
    out["purchase_14d"] = out.order_count_14d.gt(0).astype(int)
    out["ncm_14d"] = out.net_revenue_14d - out.cogs_14d - out.shipping_cost_14d - out.refund_amount_applicable
    out["ncm_14d_all_returns"] = out.net_revenue_14d - out.cogs_14d - out.shipping_cost_14d - out.total_linked_refund_amount
    out["aov_14d"] = (out.net_revenue_14d / out.order_count_14d).where(out.order_count_14d.gt(0))
    for column in ["order_count_14d", "offer_redeemed_14d", "in_window_return_count", "post_window_return_count"]:
        out[column] = out[column].astype(int)
    out = out.drop(columns="offer_type").reset_index()
    if out.order_count_14d.sum() != len(eligible) or not np.isclose(out.net_revenue_14d.sum(), eligible.net_revenue.sum()):
        raise AssertionError("Outcome aggregation failed to reconcile with eligible orders")
    out.attrs["definition"] = dict(outcome_days=14, primary_refunds="in_window", free_shipping_cutoff=free_shipping_cutoff,
        standard_shipping_fee=standard_shipping_fee, as_of_ts=str(as_of_ts) if as_of_ts is not None else None,
        late_refunds="Retrospective snapshot sensitivity, not labels available at earlier fit dates")
    return out
