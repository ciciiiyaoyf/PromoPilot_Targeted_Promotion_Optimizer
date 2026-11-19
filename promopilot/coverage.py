"""Distinguish observed transactions from verified complete purchase history.

The last observed transaction is a diagnostic, never a data-coverage watermark.
Recent counts remain counts of supplied completed orders. Their complete-history
counterparts are unknown unless an explicit, already-available source contract
certifies all transaction sources for the entire requested window.
"""
import numpy as np
import pandas as pd


from .features import _decision_spine
WINDOWS = (30, 90)
ORGANIC_SOURCE = "organic_preperiod"
CAMPAIGN_SOURCE = "campaign_attributed"


def _timestamp(value, label):
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"{label} requires an explicit valid timestamp")
    return result


def _orders(orders):
    required = {"order_id", "customer_id", "order_ts", "order_source", "is_completed"}
    if not required.issubset(orders):
        raise ValueError(f"Missing cleaned-order columns: {sorted(required-set(orders))}")
    if orders[list(required)].isna().any().any() or not orders.order_id.is_unique:
        raise ValueError("Orders require unique IDs and non-null transaction inputs")
    if not orders.is_completed.isin([True, False]).all():
        raise ValueError("is_completed must contain boolean or binary flags")
    result = orders.copy()
    result["order_ts"] = pd.to_datetime(result.order_ts, errors="raise")
    return result


def _intervals(manifest):
    if manifest is None:
        manifest = {}
    if not isinstance(manifest, dict) or not isinstance(manifest.get("coverage_intervals", []), list):
        raise ValueError("Coverage manifest must contain a coverage_intervals list")
    intervals = []
    for entry in manifest.get("coverage_intervals", []):
        if not isinstance(entry, dict):
            raise ValueError("Each coverage interval must be an object")
        required = {"scope", "start", "end", "available_as_of"}
        if not required.issubset(entry):
            raise ValueError(f"Missing coverage interval fields: {sorted(required-set(entry))}")
        if entry["scope"] != "all_sources":
            raise ValueError("Only explicit all_sources coverage can certify complete purchase history")
        start = _timestamp(entry["start"], "coverage start")
        end = _timestamp(entry["end"], "coverage end")
        available = _timestamp(entry["available_as_of"], "coverage available_as_of")
        if end <= start:
            raise ValueError("Coverage end must be later than coverage start")
        end_inclusive = entry.get("end_inclusive", False)
        if not isinstance(end_inclusive, bool):
            raise ValueError("Coverage end_inclusive must be boolean")
        ids = entry.get("customer_ids")
        if ids is not None and (not isinstance(ids, list) or not ids or any(pd.isna(cid) for cid in ids)):
            raise ValueError("customer_ids must be a nonempty ID list or null for global coverage")
        intervals.append(dict(start=start, end=end, end_inclusive=end_inclusive,
                              available_as_of=available, customer_ids=None if ids is None else set(ids)))
    return intervals


def _window_is_covered(intervals, customer_id, start, end, end_inclusive):
    """Cover the entire window with the union of eligible explicit intervals.

    Starts are inclusive. End inclusivity matters when a contract ends exactly
    at a decision cutoff. Adjacent intervals may jointly certify one window.
    A later-issued certification cannot alter historical feature availability.
    """
    eligible = [interval for interval in intervals
                if interval["available_as_of"] <= end and
                (interval["customer_ids"] is None or customer_id in interval["customer_ids"])]
    cursor = start
    covered_endpoint = False
    started = False
    for interval in sorted(eligible, key=lambda row: row["start"]):
        left, right = interval["start"], interval["end"]
        if right < start or (right == start and not interval["end_inclusive"]):
            continue
        if left > cursor:
            break
        started = True
        if right > cursor:
            cursor, covered_endpoint = right, interval["end_inclusive"]
        elif right == cursor:
            covered_endpoint = covered_endpoint or interval["end_inclusive"]
        if cursor > end or (cursor == end and (not end_inclusive or covered_endpoint)):
            return True
    return bool(started and (cursor > end or (cursor == end and (not end_inclusive or covered_endpoint))))


def build_transaction_coverage(spine, orders, key="assignment_id", cut="send_ts", inclusive=False,
                               manifest=None):
    """Return one diagnostic row per decision, without changing model features.

    Requested windows begin at cutoff minus 30/90 days (inclusive). Historical
    windows exclude the decision instant; scoring can explicitly include it.
    ``observed_*`` counts refer only to supplied completed orders. The
    ``complete_orders_*`` fields are NaN unless eligible all-source coverage
    intervals cover the entire window. ``latest_observed_organic_order_ts`` is
    customer-specific; ``source_latest_observed_organic_order_ts`` describes the
    whole supplied feed as known by that decision. Neither certifies coverage.
    """
    if not isinstance(inclusive, (bool, np.bool_)):
        raise ValueError("inclusive must be boolean")
    decisions, transactions = _decision_spine(spine, key, cut), _orders(orders)
    intervals = _intervals(manifest)
    result = decisions.set_index(key).copy()
    result["window_end_inclusive"] = bool(inclusive)
    joined = decisions.merge(transactions[["customer_id", "order_ts", "order_source", "is_completed"]],
                             on="customer_id", how="inner", validate="m:m")
    prior = joined.loc[(joined.order_ts <= joined[cut]) if inclusive else (joined.order_ts < joined[cut])].copy()
    completed = prior.loc[prior.is_completed.eq(True)]
    for days in WINDOWS:
        result[f"window_{days}d_start"] = result[cut] - pd.Timedelta(days=days)
        result[f"window_{days}d_end"] = result[cut]
        window = completed.loc[completed.order_ts >= completed[cut]-pd.Timedelta(days=days)]
        counts = window.groupby(key).size().reindex(result.index, fill_value=0).astype("int64")
        result[f"observed_orders_{days}d"] = counts
        for source, label in [(ORGANIC_SOURCE, "organic"), (CAMPAIGN_SOURCE, "campaign")]:
            result[f"observed_{label}_orders_{days}d"] = window.loc[window.order_source.eq(source)].groupby(
                key).size().reindex(result.index, fill_value=0).astype("int64")
        verified = np.array([_window_is_covered(intervals, customer_id,
                            cutoff-pd.Timedelta(days=days), cutoff, bool(inclusive))
                             for customer_id, cutoff in zip(decisions.customer_id, decisions[cut])], dtype=bool)
        result[f"orders_{days}d_coverage_status"] = np.where(verified, "verified_complete", "unverified")
        result[f"complete_orders_{days}d"] = np.where(verified, counts.to_numpy(float), np.nan)
    organic = prior.loc[prior.order_source.eq(ORGANIC_SOURCE)]
    result["latest_observed_organic_order_ts"] = organic.groupby(key).order_ts.max()
    all_organic = transactions.loc[transactions.order_source.eq(ORGANIC_SOURCE), "order_ts"]
    latest = {}
    for cutoff in decisions[cut].unique():
        timestamp = pd.Timestamp(cutoff)
        available = all_organic[all_organic <= timestamp] if inclusive else all_organic[all_organic < timestamp]
        latest[timestamp] = available.max()
    result["source_latest_observed_organic_order_ts"] = result[cut].map(latest)
    any_verified = result[[f"orders_{days}d_coverage_status" for days in WINDOWS]].eq("verified_complete").any(axis=1)
    result["coverage_interpretation"] = np.where(
        any_verified,
        "complete counts are certified only for windows marked verified_complete; consult each window's status",
        "observed records are not verified complete transaction history",
    )
    return result.reset_index()
