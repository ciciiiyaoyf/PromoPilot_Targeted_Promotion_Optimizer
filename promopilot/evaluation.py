"""Predictive diagnostics and logged-policy evaluation.

Doubly robust (DR) estimates assume unconditional randomized offers within each
campaign. Empirical logging probabilities are re-estimated in every customer
bootstrap; uniform assignment is reported as a sensitivity. Intervals condition
on fitted models and fixed allocations, and do not describe future campaign drift.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)

from .config import ACTIONS


def _purchase_arrays(y, probability):
    y, probability = np.asarray(y), np.asarray(probability, dtype=float)
    if y.ndim != 1 or probability.shape != y.shape or not len(y):
        raise ValueError("Labels and probabilities must be nonempty matching one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Labels must be binary and probabilities finite in [0, 1]")
    return y.astype(int), probability


def select_threshold(y, probability):
    """Freeze a maximum-F1 threshold from development predictions only.

    Equal-F1 ties select the lowest threshold. This classification threshold is
    a diagnostic operating point; promotion allocation maximizes NCM separately.
    """
    y, probability = _purchase_arrays(y, probability)
    if not y.any():
        raise ValueError("Threshold selection requires observed development purchasers")
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[np.argmax(f1)])


def classification_metrics(y, probability, threshold):
    """Ranking and probability quality, plus recall at a frozen threshold."""
    y, probability = _purchase_arrays(y, probability)
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("A frozen threshold must be finite in [0, 1]")
    positives = int(y.sum())
    predicted = probability >= threshold
    tp = int(((y == 1) & predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = positives - tp
    tn = len(y) - positives - fp
    return {
        "n": len(y), "positives": positives,
        "average_precision": float(average_precision_score(y, probability)) if positives else np.nan,
        "recall": tp / positives if positives else np.nan,
        "precision": tp / (tp + fp) if tp + fp else 0.,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
        "roc_auc": float(roc_auc_score(y, probability)) if 0 < positives < len(y) else np.nan,
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "observed_mean": float(y.mean()), "predicted_mean": float(probability.mean()),
        "threshold": threshold, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "predicted_positive_share": float(predicted.mean()),
    }


def regression_metrics(y, prediction):
    """Errors in dollars; positive mean/total error means overprediction.

    WAPE and relative total error divide by sum(abs(actual)), which remains
    meaningful with signed NCM. They are undefined for an all-zero target.
    R-squared is undefined for constant targets. Ratios are fractions, not percent.
    """
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.ndim != 1 or prediction.shape != y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Regression metrics require matching finite nonempty one-dimensional arrays")
    error = prediction - y
    sst = ((y - y.mean()) ** 2).sum()
    scale = np.abs(y).sum()
    return {
        "n": len(y), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.abs(error).mean()),
        "r2": float(1 - np.sum(error ** 2) / sst) if sst > 0 else np.nan,
        "mean_error": float(error.mean()), "total_error": float(error.sum()),
        "wape": float(np.abs(error).sum() / scale) if scale > 0 else np.nan,
        "relative_total_error": float(error.sum() / scale) if scale > 0 else np.nan,
        "obs_mean": float(y.mean()), "pred_mean": float(prediction.mean()),
    }


def cluster_dr_interval(frame, direct_column="direct_component", residual_column="residual_component",
                        included=None, alpha=0.05, draws=2000, seed=42):
    """Bootstrap a DR mean while re-estimating each campaign's logging policy.

    Each row contributes ``direct + residual / p(observed_action | campaign)``.
    Sample customers with all their campaign observations. The optional boolean
    ``included`` mask limits the outcome mean (e.g. a cold-start subgroup), but
    logging probabilities always use the entire resampled population supplied
    in ``frame``. Pass one row per decision, not stacked competing policies.

    Draws lacking an observed treatment arm in a represented campaign, or lacking
    subgroup observations, cannot identify the requested estimate. They are
    skipped; if too many draws fail support, return undefined interval endpoints
    rather than claim precision from a heavily conditioned bootstrap.
    """
    required = {"customer_id", "campaign_id", "observed_action", direct_column, residual_column}
    if not required.issubset(frame):
        raise ValueError(f"Missing DR bootstrap columns: {sorted(required - set(frame))}")
    if frame.empty or frame[["customer_id", "campaign_id", "observed_action"]].isna().any().any():
        raise ValueError("DR bootstrap requires nonempty rows and non-null identifiers/actions")
    if "assignment_id" in frame and not frame.assignment_id.is_unique:
        raise ValueError("DR bootstrap requires one row per decision, not stacked policies")
    if not frame.observed_action.isin(ACTIONS).all() or not 0 < alpha < 1 or draws < 2:
        raise ValueError("Invalid logging actions or interval parameters")
    direct = frame[direct_column].to_numpy(float)
    residual = frame[residual_column].to_numpy(float)
    if not np.isfinite(direct).all() or not np.isfinite(residual).all():
        raise ValueError("DR bootstrap components must be finite")
    mask = np.ones(len(frame), dtype=bool) if included is None else np.asarray(included)
    if mask.shape != (len(frame),) or mask.dtype.kind != "b" or not mask.any():
        raise ValueError("included must be a nonempty matching boolean mask")
    _, customer = np.unique(frame.customer_id.to_numpy(), return_inverse=True)
    _, campaign = np.unique(frame.campaign_id.to_numpy(), return_inverse=True)
    action = frame.observed_action.map({a: j for j, a in enumerate(ACTIONS)}).to_numpy()
    k, campaigns = customer.max() + 1, campaign.max() + 1
    groups = campaigns * len(ACTIONS)
    group = campaign * len(ACTIONS) + action
    # Aggregate once, then each sampled customer retains every decision row.
    cell = customer * groups + group
    counts = np.bincount(cell, minlength=k * groups).reshape(k, groups)
    residuals = np.bincount(cell[mask], weights=residual[mask], minlength=k * groups).reshape(k, groups)
    directs = np.bincount(customer[mask], weights=direct[mask], minlength=k)
    included_counts = np.bincount(customer[mask], minlength=k)
    observed_counts = counts.sum(axis=0).reshape(campaigns, len(ACTIONS))
    if (observed_counts == 0).any():
        raise ValueError("Every evaluation campaign needs positive support for all four actions")
    inverse_p = np.repeat(observed_counts.sum(axis=1), len(ACTIONS)) / observed_counts.ravel()
    point = float((directs.sum() + residuals.sum(axis=0) @ inverse_p) / included_counts.sum())
    if k < 2:
        return point, np.nan, np.nan
    if np.all(direct[mask] == direct[mask][0]) and np.all(residual[mask] == 0):
        return point, point, point
    rng = np.random.default_rng(seed)
    samples = []
    attempted = 0
    valid_count = 0
    while valid_count < draws and attempted < draws * 10:
        batch = min(100, draws - valid_count)
        idx = rng.integers(0, k, size=(batch, k))
        action_counts = counts[idx].sum(axis=1).reshape(batch, campaigns, len(ACTIONS))
        campaign_counts = action_counts.sum(axis=2)
        denominator = included_counts[idx].sum(axis=1)
        supported = (((action_counts > 0).all(axis=2)) | (campaign_counts == 0)).all(axis=1)
        supported &= denominator > 0
        inverse = np.divide(np.repeat(campaign_counts, len(ACTIONS), axis=1), action_counts.reshape(batch, groups),
                            out=np.zeros((batch, groups), dtype=float), where=action_counts.reshape(batch, groups) > 0)
        numerator = directs[idx].sum(axis=1) + (residuals[idx].sum(axis=1) * inverse).sum(axis=1)
        if supported.any():
            samples.append(numerator[supported] / denominator[supported])
            valid_count += int(supported.sum())
        attempted += batch
    # With the supplied campaign sizes, arm disappearance is negligibly rare.
    # This guard avoids spuriously tight intervals in tiny unsupported examples.
    if valid_count < draws or valid_count / attempted < 0.8:
        return point, np.nan, np.nan
    low, high = np.quantile(np.concatenate(samples)[:draws], [alpha / 2, 1 - alpha / 2])
    return point, float(low), float(high)


def dr_components(y, observed_action, policy_action, mu, propensity):
    """Per-row DR reward score; all four potential-outcome predictions are required."""
    y, mu, p = np.asarray(y, float), np.asarray(mu, float), np.asarray(propensity, float)
    observed_action, policy_action = np.asarray(observed_action), np.asarray(policy_action)
    n = len(y)
    if y.ndim != 1 or observed_action.shape != (n,) or policy_action.shape != (n,):
        raise ValueError("Outcomes and both action arrays must have matching one-dimensional shapes")
    if mu.shape != (n, len(ACTIONS)) or p.shape != (n,):
        raise ValueError("Prediction/propensity dimensions do not match logged outcomes")
    if not np.isfinite(mu).all() or not np.isfinite(y).all() or not np.isfinite(p).all() or (p <= 0).any() or (p > 1).any():
        raise ValueError("Finite outcomes and positive logging probabilities are required")
    if not set(observed_action).issubset(ACTIONS) or not set(policy_action).issubset(ACTIONS):
        raise ValueError("Invalid action")
    observed = np.array([ACTIONS.index(a) for a in observed_action])
    policy = np.array([ACTIONS.index(a) for a in policy_action])
    weights = (observed_action == policy_action).astype(float) / p
    phi = mu[np.arange(n), policy] + weights * (y - mu[np.arange(n), observed])
    return phi, weights


def evaluate_policy(data, allocation, mu, cost_mu, policy_name, split, draws=2000):
    required = {"assignment_id", "customer_id", "campaign_id", "offer_type", "ncm_14d", "promo_cost_14d"}
    if not required.issubset(data) or not {"customer_id", "recommended_action", "estimated_promo_cost"}.issubset(allocation):
        raise ValueError("Missing evaluation identifiers, outcomes, or allocation columns")
    if data.empty or data[["assignment_id", "customer_id", "campaign_id"]].isna().any().any() or not data.assignment_id.is_unique:
        raise ValueError("Evaluation requires nonempty unique assignment IDs and non-null customer/campaign IDs")
    if allocation.customer_id.isna().any() or not data.customer_id.is_unique or not allocation.customer_id.is_unique:
        raise ValueError("One decision per customer required within an evaluation campaign")
    if set(data.customer_id) != set(allocation.customer_id):
        raise ValueError("Policy must cover exactly the evaluation population")
    if data.campaign_id.nunique() != 1:
        raise ValueError("evaluate_policy expects one campaign; pool decomposed scores only after evaluation")
    chosen = allocation.set_index("customer_id").reindex(data.customer_id)
    if chosen.recommended_action.isna().any():
        raise ValueError("Policy does not cover the evaluation population")
    if "assignment_id" in chosen and not np.array_equal(chosen.assignment_id.to_numpy(), data.assignment_id.to_numpy()):
        raise ValueError("Allocation assignment IDs do not match evaluation customers")
    estimated_cost = chosen.estimated_promo_cost.to_numpy(float)
    if not np.isfinite(estimated_cost).all() or (estimated_cost < 0).any():
        raise ValueError("Estimated promotion costs must be finite and nonnegative")
    observed = data.offer_type.to_numpy()
    policy = chosen.recommended_action.to_numpy()
    # No logging probabilities are supplied. Empirical frequencies are valid only
    # under unconditional randomization within campaign; uniform assignment is a sensitivity.
    counts = data.offer_type.value_counts(normalize=True)
    if set(counts.index) != set(ACTIONS):
        raise ValueError("All four actions need positive evaluation support")
    p = data.offer_type.map(counts).to_numpy()
    y = data.ncm_14d.to_numpy(float)
    no = np.full(len(data), "NO_OFFER")
    phi, w = dr_components(y, observed, policy, mu, p)
    phi0, w0 = dr_components(y, observed, no, mu, p)
    delta = phi - phi0
    observed_index = np.array([ACTIONS.index(a) for a in observed])
    policy_index = np.array([ACTIONS.index(a) for a in policy])
    observed_mu = np.asarray(mu)[np.arange(len(data)), observed_index]
    individual = data[["assignment_id", "customer_id", "campaign_id"]].copy()
    individual["policy"] = policy_name
    individual["recommended_action"] = policy
    individual["outcome"] = y
    individual["observed_action"] = observed
    individual["direct_component"] = np.asarray(mu)[np.arange(len(data)), policy_index] - np.asarray(mu)[:, 0]
    individual["residual_component"] = ((observed == policy).astype(float) - (observed == no).astype(float)) * (y - observed_mu)
    val, low, high = cluster_dr_interval(individual, draws=draws)
    if not np.isclose(val, delta.mean()):
        raise AssertionError("DR decomposition does not reproduce the point estimate")
    cost_phi, _ = dr_components(data.promo_cost_14d, observed, policy, cost_mu, p)
    cost_components = individual[["assignment_id", "customer_id", "campaign_id", "observed_action"]].copy()
    cost_components["direct_component"] = np.asarray(cost_mu)[np.arange(len(data)), policy_index]
    cost_components["residual_component"] = (observed == policy).astype(float) * (
        data.promo_cost_14d.to_numpy(float) - np.asarray(cost_mu)[np.arange(len(data)), observed_index])
    cost_value, cost_low, cost_high = cluster_dr_interval(cost_components, draws=draws)
    if not np.isclose(cost_value, cost_phi.mean()):
        raise AssertionError("Cost DR decomposition does not reproduce the point estimate")
    uniform, _ = dr_components(y, observed, policy, mu, np.full(len(data), 0.25))
    uniform0, _ = dr_components(y, observed, no, mu, np.full(len(data), 0.25))
    hajek = np.sum(w*y)/np.sum(w) - np.sum(w0*y)/np.sum(w0) if w.sum() and w0.sum() else np.nan
    summary = {"split": split, "policy": policy_name, "n": len(data),
               "dr_ncm_per_customer": float(phi.mean()), "dr_incremental_ncm_per_customer": val,
               "ci_low": low, "ci_high": high, "hajek_incremental_ncm_per_customer": float(hajek),
               "uniform_propensity_dr_incremental_ncm": float(np.mean(uniform-uniform0)),
               "estimated_promo_spend": float(chosen.estimated_promo_cost.sum()),
               "dr_promo_spend": cost_value*len(data), "dr_promo_spend_ci_low": cost_low*len(data),
               "dr_promo_spend_ci_high": cost_high*len(data),
               "offered_share": float(np.mean(policy != "NO_OFFER")),
               "matched_rows": int(np.sum(observed == policy)),
               "effective_sample_size": float(w.sum()**2 / np.sum(w*w)) if np.sum(w*w) else 0.,
               "propensity_source": "empirical_within_campaign_unconditional_randomization_assumed",
               "interval_method": "customer_cluster_bootstrap_reestimating_campaign_logging_frequencies"}
    individual["dr_score"] = phi
    individual["dr_no_offer_score"] = phi0
    individual["dr_incremental_score"] = delta
    individual["matched"] = observed == policy
    return summary, individual


def value_table(data, mu, costs, purchase_probability):
    """Construct four action values in the supplied customer row order.

    Prediction rows must correspond to data rows. Historical send timestamps
    become policy cutoffs; July's explicit policy_as_of_ts is preserved.
    """
    timestamp = "policy_as_of_ts" if "policy_as_of_ts" in data else "send_ts"
    if not {"customer_id", "campaign_id", timestamp}.issubset(data) or data.empty:
        raise ValueError("Value tables need customer, campaign and decision timestamps")
    if data[["customer_id", "campaign_id", timestamp]].isna().any().any() or not data.customer_id.is_unique:
        raise ValueError("Value tables require unique non-null customers and decision context")
    mu, costs = np.asarray(mu, float), np.asarray(costs, float)
    probability = np.asarray(purchase_probability, float)
    if mu.shape != (len(data), len(ACTIONS)) or costs.shape != mu.shape or probability.shape != (len(data),):
        raise ValueError("Prediction arrays do not match the customer/action rows")
    if not np.isfinite(mu).all() or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Action values must be finite and costs nonnegative")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Purchase probabilities must be finite in [0, 1]")
    context = data[["customer_id", "campaign_id", timestamp]].rename(columns={timestamp: "policy_as_of_ts"})
    return pd.concat([
        context.assign(action=action, predicted_ncm=mu[:, j],
                       estimated_incremental_ncm=mu[:, j] - mu[:, 0],
                       estimated_promo_cost=costs[:, j],
                       predicted_purchase_probability=probability)
        for j, action in enumerate(ACTIONS)
    ], ignore_index=True)
