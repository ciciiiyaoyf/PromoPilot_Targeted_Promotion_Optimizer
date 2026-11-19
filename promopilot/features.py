"""Customer features available at each campaign send or policy scoring cutoff.

Cutoff rule: historical rows use events with ts < send_ts (inclusive=False);
July scoring uses ts <= policy_as_of_ts (inclusive=True).

Purchase, return and product history describe events observed in the supplied
extract. Their ``observed_`` names do not claim complete transaction coverage:
an observed zero is not evidence that the customer made no purchases.
"""
import numpy as np
import pandas as pd

PROFILE_CAT = ["region", "acquisition_channel", "customer_tier", "preferred_category", "shipping_zone"]
PROFILE_NUM = ["customer_tenure_days", "is_member", "marketing_consent"]
FAMILIES = {
    "profile": PROFILE_NUM + PROFILE_CAT,
    "purchase": ["observed_prior_order_count", "observed_orders_30d", "observed_orders_90d", "observed_days_since_last_order", "observed_gross_revenue_90d",
                 "observed_net_revenue_90d", "observed_avg_order_value_90d", "observed_discount_amount_90d", "observed_cogs_90d", "observed_shipping_revenue_90d",
                 "observed_shipping_cost_90d", "observed_historical_ncm_90d", "prior_offer_redemption_rate"],
    "returns": ["observed_returns_90d", "observed_refund_amount_90d", "observed_historical_return_rate"],
    "web": ["sessions_7d", "sessions_30d", "pages_viewed_30d", "avg_pages_per_session_30d", "cart_add_sessions_30d",
            "cart_add_rate_30d", "days_since_last_session", "top_category_viewed_30d"],
    "email": ["prior_emails_delivered", "prior_email_opens", "prior_email_clicks", "historical_open_rate",
              "historical_click_rate", "days_since_last_email_open", "days_since_last_email_click"],
    "product": ["observed_units_purchased_90d", "observed_distinct_products_90d", "observed_distinct_categories_90d", "observed_top_purchase_category_90d"],
}
CATEGORICAL = PROFILE_CAT + ["top_category_viewed_30d", "observed_top_purchase_category_90d"]
FEATURES = [feature for family in FAMILIES.values() for feature in family]


def _decision_spine(spine, key, cut):
    """Validate decision identity and parse timestamps without modifying inputs."""
    columns = list(dict.fromkeys([key, "customer_id", cut]))
    if not set(columns).issubset(spine):
        raise ValueError("Decisions require a key, customer_id, and cutoff timestamp")
    result = spine[columns].copy()
    result[cut] = pd.to_datetime(result[cut], errors="raise")
    if result[columns].isna().any().any() or not result[key].is_unique:
        raise ValueError("Decision keys must be unique and identifiers/cutoffs non-null")
    return result


def _prior(spine, ev, ts, key, cut, inclusive, audit, family):
    """Join events to spine rows by customer and keep only events known at the row cutoff."""
    m = spine[list(dict.fromkeys([key, "customer_id", cut]))].merge(ev, on="customer_id")
    m = m[(m[ts] <= m[cut]) if inclusive else (m[ts] < m[cut])].copy()
    m["_age_d"] = (m[cut] - m[ts]).dt.total_seconds() / 86400
    lag = (m[ts] - m[cut]).max() if len(m) else pd.Timedelta(-1, "D")
    audit[family] = max(audit.get(family, lag), lag)
    return m


def _top(df, key, cat, weight=None):
    g = df.groupby([key, cat]).size() if weight is None else df.groupby([key, cat])[weight].sum()
    g = g.rename("_n").reset_index().sort_values([key, "_n", cat], ascending=[True, False, True])
    return g.drop_duplicates(key).set_index(key)[cat]


def _div(a, b):
    return (a / b.where(b > 0)).astype(float)


def build_features(spine, t, key="assignment_id", cut="send_ts", inclusive=False):
    """spine: one row per decision (key, customer_id, cut). t: validated, cleaned source tables.
    Returns (features DataFrame with one row per key, audit dict of max event_ts - cutoff per family).

    Counts and sums of extract-observed transactions are zero when no supplied
    events fall in the original decision-relative window. Coverage-qualified
    counts and coverage metadata are separate from these 43 model predictors.
    """
    if not isinstance(inclusive, (bool, np.bool_)):
        raise ValueError("inclusive must be boolean")
    audit = {}
    s = _decision_spine(spine, key, cut)
    out = s.set_index(key)
    w = lambda m, d: m[m._age_d <= d]
    # Profile
    cu = t["customers"].set_index("customer_id")
    if not cu.index.is_unique or not s.customer_id.isin(cu.index).all():
        raise ValueError("Each decision must reference one known customer")
    prof = s.join(cu, on="customer_id")
    if prof.signup_date.isna().any() or prof.signup_date.gt(prof[cut]).any():
        raise ValueError("Customer signup must be known by the decision cutoff")
    out["customer_tenure_days"] = ((prof[cut] - prof.signup_date).dt.total_seconds() / 86400).values
    for c in ["is_member", "marketing_consent"] + PROFILE_CAT:
        out[c] = prof[c].values
    # Purchase history (completed orders only)
    orders = t["orders"][t["orders"].is_completed]
    po = _prior(s, orders, "order_ts", key, cut, inclusive, audit, "purchase")
    rets = t["returns"][["order_id", "return_ts", "refund_amount"]]
    pr = po[[key, "order_id", cut]].merge(rets, on="order_id")
    pr = pr[(pr.return_ts <= pr[cut]) if inclusive else (pr.return_ts < pr[cut])]
    if len(pr):
        audit["purchase_refunds"] = (pr.return_ts - pr[cut]).max()
    # Multiple partial refunds must not multiply the original order or its items.
    known_refunds = pr.groupby([key, "order_id"], as_index=False).refund_amount.sum()
    po = po.merge(known_refunds, on=[key, "order_id"], how="left", validate="1:1")
    po["known_refund"] = po.refund_amount.fillna(0)
    po["ncm"] = po.net_revenue - po.cogs - po.shipping_cost - po.known_refund
    g_all = po.groupby(key)
    out["observed_prior_order_count"] = g_all.size()
    out["observed_orders_30d"] = w(po, 30).groupby(key).size()
    p90 = w(po, 90).groupby(key)
    out["observed_orders_90d"] = p90.size()
    out["observed_days_since_last_order"] = g_all._age_d.min()
    for src, dst in [("merchandise_gross_revenue", "observed_gross_revenue_90d"), ("net_revenue", "observed_net_revenue_90d"),
                     ("discount_amount", "observed_discount_amount_90d"), ("cogs", "observed_cogs_90d"),
                     ("shipping_revenue", "observed_shipping_revenue_90d"), ("shipping_cost", "observed_shipping_cost_90d"),
                     ("ncm", "observed_historical_ncm_90d")]:
        out[dst] = p90[src].sum()
    # Prior offer redemption: prior commercial offers (excl. current) redeemed by a completed order known at cutoff
    ha = t["historical_assignments"]
    pa = _prior(s, ha.rename(columns={"send_ts": "prior_send_ts", "assignment_id": "prior_assignment_id"}),
                "prior_send_ts", key, cut, inclusive, audit, "prior_assignments")
    if key == "assignment_id":
        pa = pa[pa.prior_assignment_id != pa[key]]
    off = pa[pa.offer_type != "NO_OFFER"]
    red = orders[orders.offer_redeemed == 1][["customer_id", "campaign_id", "order_ts"]]
    off = off.merge(red, on=["customer_id", "campaign_id"], how="left")
    off["redeemed"] = ((off.order_ts <= off[cut]) if inclusive else (off.order_ts < off[cut])).astype(int)
    off = off.groupby([key, "prior_assignment_id"]).redeemed.max().reset_index()
    out["prior_offer_redemption_rate"] = off.groupby(key).redeemed.mean()
    # Returns history (return events known at cutoff)
    rt = rets.merge(t["orders"][["order_id", "customer_id"]], on="order_id")
    pre = _prior(s, rt, "return_ts", key, cut, inclusive, audit, "returns")
    out["observed_returns_90d"] = w(pre, 90).groupby(key).size()
    out["observed_refund_amount_90d"] = w(pre, 90).groupby(key).refund_amount.sum()
    n_ret = po[po.known_refund > 0].groupby(key).order_id.nunique().reindex(out.index, fill_value=0)
    out["observed_historical_return_rate"] = _div(n_ret, out.observed_prior_order_count.fillna(0))
    # Web behaviour (identified sessions only)
    ws = t["web_sessions"][t["web_sessions"].customer_id.notna()]
    pw = _prior(s, ws, "session_ts", key, cut, inclusive, audit, "web")
    w30 = w(pw, 30)
    out["sessions_7d"] = w(pw, 7).groupby(key).size()
    out["sessions_30d"] = w30.groupby(key).size()
    out["pages_viewed_30d"] = w30.groupby(key).pages_viewed.sum()
    out["cart_add_sessions_30d"] = w30.groupby(key).added_to_cart.sum()
    out["days_since_last_session"] = pw.groupby(key)._age_d.min()
    out["top_category_viewed_30d"] = _top(w30, key, "category_viewed")
    # Email engagement (delivered prior campaign emails + opens/clicks known at cutoff)
    out["prior_emails_delivered"] = pa[pa.email_delivered == 1].groupby(key).size()
    pe = _prior(s, t["email_events"].drop(columns=["assignment_id", "campaign_id"]), "event_ts", key, cut, inclusive,
                audit, "email")
    for et, n in [("open", "opens"), ("click", "clicks")]:
        e = pe[pe.event_type == et].groupby(key)
        out[f"prior_email_{n}"] = e.size()
        out[f"days_since_last_email_{et}"] = e._age_d.min()
    # Product / category history (deduplicated items of completed prior-90d orders)
    items = t["order_items"].merge(t["products"][["product_id", "category"]], on="product_id")
    pi = w(po, 90)[[key, "order_id"]].merge(items, on="order_id")
    gi = pi.groupby(key)
    out["observed_units_purchased_90d"] = gi.quantity.sum()
    out["observed_distinct_products_90d"] = gi.product_id.nunique()
    out["observed_distinct_categories_90d"] = gi.category.nunique()
    out["observed_top_purchase_category_90d"] = _top(pi, key, "category", "quantity")
    # Fill counts/sums with 0; rates/recency stay missing when undefined
    zero = ["observed_prior_order_count", "observed_orders_30d", "observed_orders_90d", "observed_gross_revenue_90d", "observed_net_revenue_90d",
            "observed_discount_amount_90d", "observed_cogs_90d", "observed_shipping_revenue_90d", "observed_shipping_cost_90d", "observed_historical_ncm_90d",
            "observed_returns_90d", "observed_refund_amount_90d", "sessions_7d", "sessions_30d", "pages_viewed_30d",
            "cart_add_sessions_30d", "prior_emails_delivered", "prior_email_opens", "prior_email_clicks",
            "observed_units_purchased_90d", "observed_distinct_products_90d", "observed_distinct_categories_90d"]
    out[zero] = out[zero].fillna(0)
    out["observed_avg_order_value_90d"] = _div(out.observed_net_revenue_90d, out.observed_orders_90d)
    out["avg_pages_per_session_30d"] = _div(out.pages_viewed_30d, out.sessions_30d)
    out["cart_add_rate_30d"] = _div(out.cart_add_sessions_30d, out.sessions_30d)
    out["historical_open_rate"] = _div(out.prior_email_opens, out.prior_emails_delivered)
    out["historical_click_rate"] = _div(out.prior_email_clicks, out.prior_emails_delivered)
    result = out.reset_index()[list(dict.fromkeys([key, "customer_id", cut])) + FEATURES]
    return result, audit
