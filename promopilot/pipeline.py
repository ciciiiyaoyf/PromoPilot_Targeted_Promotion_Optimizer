"""One reproducible path from raw extracts to evaluated, constrained assignments."""
from dataclasses import asdict
from pathlib import Path
import json
import platform
from importlib import import_module

import numpy as np
import pandas as pd

from .allocation import (optimize_policy, get_holdout, measurement_assignment,
                         build_baseline_policies, assert_feasible)
from .config import ACTIONS, PipelineConfig
from .coverage import build_transaction_coverage
from .data import load_tables, validate_and_clean
from .evaluation import (classification_metrics, regression_metrics, select_threshold,
                         value_table, evaluate_policy)
from .features import FEATURES, CATEGORICAL, build_features
from .models import (fit_logistic_baseline, fit_profit_models, predict_actions,
                     fit_cost_models, predict_costs, fit_nuisance_models)
from .outcomes import build_outcomes

NUMERIC = [feature for feature in FEATURES if feature not in CATEGORICAL]


def training_before(history, cutoff):
    """Require complete 14-day labels before fitting a historical decision."""
    cutoff = pd.Timestamp(cutoff)
    result = history.loc[history.send_ts.lt(cutoff) & history.outcome_end_ts.le(cutoff)].copy()
    if result.empty:
        raise ValueError(f'No mature training outcomes before {cutoff}')
    return result


def _exploration(data):
    return data.loc[data.assignment_source.eq('exploration')].copy()


def _purchase_probability(models, rows):
    # All four action views share the same fitted purchase head.
    purchase_head = models[ACTIONS[0]].purchase_model
    return purchase_head.predict_proba(rows[NUMERIC+CATEGORICAL].assign(
        _action=rows.offer_type.to_numpy()))[:, 1]


def _development_thresholds(history, config):
    evaluation = history.loc[history.campaign_id.eq(config.evaluation_campaign)]
    if evaluation.empty:
        raise ValueError('Evaluation campaign is missing')
    evaluation_cutoff = evaluation.send_ts.min()
    if not config.development_campaigns or len(set(config.development_campaigns)) != len(config.development_campaigns):
        raise ValueError('Development campaigns must be nonempty and unique')
    predictions = {'logistic': [], 'hurdle_purchase': []}
    folds = []
    for campaign in config.development_campaigns:
        print(f'Development thresholds: {campaign}', flush=True)
        valid = history.loc[history.campaign_id.eq(campaign)]
        if valid.empty:
            raise ValueError(f'Missing development campaign: {campaign}')
        if campaign == config.evaluation_campaign or not valid.outcome_end_ts.le(evaluation_cutoff).all():
            raise ValueError('Development outcomes must mature before the evaluation campaign')
        train = training_before(history, valid.send_ts.min())
        lr = fit_logistic_baseline(train, NUMERIC, CATEGORICAL)
        predictions['logistic'].append((valid.purchase_14d.to_numpy(),
            lr.predict_proba(valid[NUMERIC+CATEGORICAL])[:, 1]))
        exp_valid = _exploration(valid)
        profit = fit_profit_models(_exploration(train), NUMERIC, CATEGORICAL)
        predictions['hurdle_purchase'].append((exp_valid.purchase_14d.to_numpy(),
            _purchase_probability(profit, exp_valid)))
        folds.append(dict(campaign=campaign, cutoff=str(valid.send_ts.min()),
            training_rows=len(train), validation_rows=len(valid),
            latest_training_outcome_end=str(train.outcome_end_ts.max())))
    thresholds = {name: select_threshold(np.concatenate([y for y, _ in items]),
        np.concatenate([p for _, p in items])) for name, items in predictions.items()}
    return thresholds, folds


def _historical_mix(history):
    return history.loc[history.assignment_source.eq('business_rule') &
        history.offer_type.ne('NO_OFFER'), 'offer_type'].value_counts(normalize=True).to_dict()


def _constraints(config, holdout, population_size, scoring_size):
    return dict(budget=config.budget*population_size/scoring_size,
        max_offer_share=config.max_offer_share, max_pct20_share=config.max_pct20_share,
        holdout_ids=holdout)


def _evaluate(history, scoring_size, thresholds, config):
    print(f'Evaluating frozen models and policies: {config.evaluation_campaign}', flush=True)
    test_all = history.loc[history.campaign_id.eq(config.evaluation_campaign)].copy()
    if test_all.empty:
        raise ValueError('Evaluation campaign is missing')
    train_all = training_before(history, test_all.send_ts.min())
    train = _exploration(train_all)
    test = _exploration(test_all).sort_values('customer_id').reset_index(drop=True)
    features = test[NUMERIC+CATEGORICAL]
    lr = fit_logistic_baseline(train_all, NUMERIC, CATEGORICAL)
    profit = fit_profit_models(train, NUMERIC, CATEGORICAL)
    costs = fit_cost_models(train, NUMERIC, CATEGORICAL, config.cost_selection_campaign)
    mu = predict_actions(profit, features)
    cost_mu = predict_costs(costs, features)
    action_index = np.array([ACTIONS.index(a) for a in test.offer_type])
    factual = mu[np.arange(len(test)), action_index]
    factual_cost = cost_mu[np.arange(len(test)), action_index]
    rows = [dict(model='logistic', task='purchase', scope='all_assignments',
        **classification_metrics(test_all.purchase_14d,
            lr.predict_proba(test_all[NUMERIC+CATEGORICAL])[:, 1], thresholds['logistic'])),
        dict(model='hurdle_purchase', task='purchase', scope='exploration',
        **classification_metrics(test.purchase_14d, _purchase_probability(profit, test),
                                 thresholds['hurdle_purchase']))]
    for scope in ['pooled'] + ACTIONS:
        mask = np.ones(len(test), dtype=bool) if scope == 'pooled' else test.offer_type.eq(scope).to_numpy()
        rows.append(dict(model='hurdle_xgb_half', task='ncm', scope=scope,
            **regression_metrics(test.loc[mask, 'ncm_14d'], factual[mask])))
        rows.append(dict(model='promotion_cost', task='cost', scope=scope,
            **regression_metrics(test.loc[mask, 'promo_cost_14d'], factual_cost[mask])))
    metrics = pd.DataFrame(rows).assign(campaign=config.evaluation_campaign)
    values = value_table(test, mu, cost_mu, lr.predict_proba(features)[:, 1])
    controls = get_holdout(test.customer_id, config.control_share, config.seed)
    limits = _constraints(config, controls, len(test), scoring_size)
    policies = {'HURDLE_XGB_HALF': optimize_policy(values, **limits)}
    policies.update(build_baseline_policies(values, historical_mix=_historical_mix(train_all),
                                            seed=config.seed, **limits))
    nuisance = predict_actions(fit_nuisance_models(train, NUMERIC, CATEGORICAL), features)
    summaries = []
    for name, allocation in policies.items():
        summary, _ = evaluate_policy(test, allocation, nuisance, cost_mu, name,
                                     config.evaluation_campaign, config.bootstrap_draws)
        summary['budget_limit'] = limits['budget']
        summaries.append(summary)
    return metrics, pd.DataFrame(summaries)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    return value


def run(data_dir, output_dir='artifacts', config=None):
    """Run from explicit raw CSV paths; generated artifacts stay out of Git."""
    config = config or PipelineConfig()
    destination = Path(output_dir).resolve()
    if destination == Path(data_dir).resolve():
        raise ValueError('Output directory must differ from the raw input directory')
    destination.mkdir(parents=True, exist_ok=True)
    print('Validating the eleven raw extracts', flush=True)
    clean, quality = validate_and_clean(load_tables(data_dir))
    quality.to_csv(destination/'data_quality.csv', index=False)
    scoring = clean['policy_scoring_population']
    if scoring.policy_as_of_ts.nunique() != 1:
        raise ValueError('This batch allocation requires one shared policy_as_of_ts')
    cutoff = scoring.policy_as_of_ts.iloc[0]
    assignments = clean['historical_assignments']
    if not (assignments.send_ts+pd.Timedelta(days=config.outcome_days)).le(cutoff).all():
        raise ValueError('Historical outcomes must mature before the scoring cutoff')
    print('Building historical and scoring features with their own time cutoffs', flush=True)
    historical_features, historical_audit = build_features(assignments, clean)
    outcomes = build_outcomes(clean, outcome_days=config.outcome_days,
        free_shipping_cutoff=config.free_shipping_cutoff,
        standard_shipping_fee=config.standard_shipping_fee, as_of_ts=cutoff)
    context = assignments.drop(columns=['customer_id', 'send_ts'])
    history = historical_features.merge(context, on='assignment_id', validate='1:1').merge(
        outcomes, on='assignment_id', validate='1:1')
    july, july_audit = build_features(scoring, clean, key='customer_id',
                                     cut='policy_as_of_ts', inclusive=True)
    july = july.merge(scoring[['customer_id', 'campaign_id']], on='customer_id', validate='1:1')
    coverage = pd.concat([
        build_transaction_coverage(assignments, clean['orders']).assign(population='historical'),
        build_transaction_coverage(scoring, clean['orders'], key='customer_id',
            cut='policy_as_of_ts', inclusive=True).assign(population='scoring')], ignore_index=True)
    coverage.to_csv(destination/'transaction_coverage.csv', index=False)
    thresholds, folds = _development_thresholds(history, config)
    metrics, policy_evaluation = _evaluate(history, len(july), thresholds, config)
    metrics.to_csv(destination/'model_metrics.csv', index=False)
    policy_evaluation.to_csv(destination/'policy_evaluation.csv', index=False)
    print('Fitting final models and solving the constrained July allocation', flush=True)
    mature = training_before(history, cutoff)
    exploration = _exploration(mature)
    features = july[NUMERIC+CATEGORICAL]
    lr = fit_logistic_baseline(mature, NUMERIC, CATEGORICAL)
    profit = fit_profit_models(exploration, NUMERIC, CATEGORICAL)
    costs = fit_cost_models(exploration, NUMERIC, CATEGORICAL, config.cost_selection_campaign)
    values = value_table(july, predict_actions(profit, features), predict_costs(costs, features),
                         lr.predict_proba(features)[:, 1])
    controls = get_holdout(july.customer_id, config.control_share, config.seed)
    limits = _constraints(config, controls, len(july), len(july))
    policy = optimize_policy(values, **limits)
    submission = policy[['customer_id', 'recommended_action', 'estimated_incremental_ncm',
                         'estimated_promo_cost']].copy()
    submission[['estimated_incremental_ncm', 'estimated_promo_cost']] = submission[
        ['estimated_incremental_ncm', 'estimated_promo_cost']].round(4)
    # Verify the actual rounded deliverable as well as the full-precision solution.
    assert_feasible(submission, expected_customers=july.customer_id, **limits)
    submission.to_csv(destination/'policy_submission.csv', index=False)
    measurement_assignment(policy).to_csv(destination/'measurement_assignments.csv', index=False)
    values.to_csv(destination/'action_values.csv', index=False)
    summary = dict(project='PromoPilot_Targeted_Promotion_Optimizer', config=asdict(config),
        environment={'python': platform.python_version(), **{p: import_module(module).__version__
            for p, module in [('numpy', 'numpy'), ('pandas', 'pandas'),
                ('scikit-learn', 'sklearn'), ('scipy', 'scipy'), ('xgboost', 'xgboost')]}},
        historical_decisions=len(history), scoring_customers=len(july), policy_as_of_ts=str(cutoff),
        feature_count=len(FEATURES), numeric_features=NUMERIC, categorical_features=CATEGORICAL,
        thresholds=thresholds, development_folds=folds,
        feature_cutoff_audit={'historical': historical_audit, 'scoring': july_audit},
        cost_model_selection={a: m.selection_ for a, m in costs.items()},
        measurement_controls=len(controls), control_design='uniform_random_seeded',
        allocation_counts={a: int(policy.recommended_action.eq(a).sum()) for a in ACTIONS},
        projected_incremental_ncm=float(policy.estimated_incremental_ncm.sum()),
        projected_promotion_cost=float(policy.estimated_promo_cost.sum()),
        solver_certificate=dict(policy.attrs),
        evaluation_status='Retrospective CP07 reanalysis; previously inspected during project development',
        coverage_status='Unverified completeness: observed order zeros do not establish zero purchases')
    summary = _json_ready(summary)
    (destination/'run_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    _write_report(destination, summary, metrics, policy_evaluation)
    print(f'Complete: {destination / "report.md"}', flush=True)
    return summary


def _write_report(destination, summary, metrics, policy_evaluation):
    lr = metrics.loc[metrics.model.eq('logistic')].iloc[0]
    ncm = metrics.loc[metrics.model.eq('hurdle_xgb_half') & metrics.scope.eq('pooled')].iloc[0]
    # Plain Markdown avoids an extra tabulate/reporting dependency.
    text = f'''# PromoPilot run

Scored **{summary['scoring_customers']:,} customers** using information available by
**{summary['policy_as_of_ts']}**.

## Historical model evaluation

CP07 is a retrospective evaluation, previously inspected during project development.
The logistic baseline uses all CP07 assignments; profit metrics use randomized exploration.

- Logistic average precision: {lr['average_precision']:.6f}; recall: {lr['recall']:.6f}.
- Logistic decision threshold: {summary['thresholds']['logistic']:.6f}, chosen using CP04–CP06.
- NCM prediction RMSE: ${ncm['rmse']:.4f}; MAE: ${ncm['mae']:.4f}; R²: {ncm['r2']:.6f}.
- Additional classification, NCM, and promotion-cost metrics: `model_metrics.csv`.
- Historical policy estimates and customer bootstrap intervals: `policy_evaluation.csv`.
  These estimates assume unconditional randomization within each exploration campaign.
  The random-free-shipping comparison uses one fixed seed, not an average over seeds.

## July model projection

- Projected incremental NCM versus all-no-offer: **${summary['projected_incremental_ncm']:,.2f}**.
- Projected promotion cost: **${summary['projected_promotion_cost']:,.2f}**.
- Reserved measurement controls: **{summary['measurement_controls']}**.
- Actions: {', '.join(f'{a} = {n:,}' for a, n in summary['allocation_counts'].items())}.

These are model projections, not realized July profit or a guaranteed causal gain.
The MILP certificate verifies optimality for these predictions and constraints.
No discount quota is imposed: an offer can legitimately receive no assignments.

## Data interpretation

Historical features use events before send time; scoring features use events at or before
the policy cutoff. Supplied recent-order counts can
be zero while complete purchase history remains unknown. See `transaction_coverage.csv`.
Primary labels count completed attributed orders and refunds inside the 14-day window.
Late refunds are excluded from that primary target. Free shipping assumes a $75 threshold
and uses the observed $5.99 customer fee below it, separately from fulfillment cost.

The four-column deliverable is `policy_submission.csv`. Preserve
`measurement_assignments.csv` for the future CONTROL-versus-POLICY comparison:
NO_OFFER customers inside POLICY are not measurement controls.
`run_summary.json` records configuration, time audits, thresholds, and solver evidence.
'''
    (destination/'report.md').write_text(text, encoding='utf-8')
