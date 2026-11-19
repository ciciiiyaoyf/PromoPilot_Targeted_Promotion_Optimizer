"""The project's three fitted components: purchase, signed profit, and cost.

Hyperparameters are fixed from the preceding project analysis. This module has
no experiment registry or filesystem state. Callers supply time-eligible rows;
the purchase baseline uses all assignments, whereas profit/cost use exploration.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .config import ACTIONS


def canonicalize_categories(values):
    """Make missing categories independent of CSV/Parquet/in-memory storage."""
    return values.astype(object).where(pd.notna(values), None)


def _categorical():
    return Pipeline([
        ('nulls', FunctionTransformer(canonicalize_categories, feature_names_out='one-to-one')),
        ('impute', SimpleImputer(strategy='constant', fill_value='Missing', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore')),
    ])


def _features(num, cat):
    columns = list(num) + list(cat)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError('Predictors must be nonempty and unique')
    if {'offer_type', '_action', 'purchase_14d', 'ncm_14d', 'promo_cost_14d'} & set(columns):
        raise ValueError('Predictors must contain customer information, not actions or outcomes')
    return columns


def _exploration(data, target):
    if data.empty or set(data.offer_type) != set(ACTIONS):
        raise ValueError('All four actions need training support')
    if 'assignment_source' not in data or not data.assignment_source.eq('exploration').all():
        raise ValueError('Profit and cost models require exploration rows only')
    if not np.isfinite(data[target].to_numpy(float)).all():
        raise ValueError('Training targets must be finite')


def fit_logistic_baseline(data, num, cat):
    """Fit the one original C=.01 purchase-propensity baseline, excluding offers."""
    columns = _features(num, cat)
    if not data.purchase_14d.isin([0, 1]).all() or data.purchase_14d.nunique() != 2:
        raise ValueError('Logistic training needs both binary purchase outcomes')
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                          ('scale', StandardScaler())]), num),
        ('cat', _categorical(), cat),
    ])
    model = Pipeline([('pre', pre), ('lr', LogisticRegression(
        C=.01, solver='lbfgs', max_iter=2000, random_state=42))])
    return model.fit(data[columns], data.purchase_14d)


def _hurdle_pipeline(num, cat, *, purchase=False):
    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), num),
        ('cat', _categorical(), cat + ['_action']),
    ])
    params = dict(n_estimators=150 if purchase else 100, max_depth=2,
                  learning_rate=.03, min_child_weight=20 if purchase else 10,
                  reg_lambda=20. if purchase else 30., subsample=.8,
                  colsample_bytree=.8, tree_method='hist', random_state=42, n_jobs=4)
    learner = (XGBClassifier(objective='binary:logistic', eval_metric='logloss', **params)
               if purchase else XGBRegressor(objective='reg:squarederror', **params))
    return Pipeline([('pre', pre), ('model', learner)])


@dataclass
class HurdleActionModel:
    """One action's purchase probability multiplied by its signed margin."""
    purchase_model: object
    margin_model: object
    action: str
    mean_margin: float

    def predict(self, features):
        action_features = features.assign(_action=self.action)
        probability = self.purchase_model.predict_proba(action_features)[:, 1]
        margin = (.5 * self.margin_model.predict(action_features)
                  + .5 * np.full(len(features), self.mean_margin))
        return np.asarray(probability * margin, dtype=float)


def fit_profit_models(data, num, cat, target='ncm_14d'):
    """Fit the selected HURDLE_XGB_HALF model on randomized exploration rows.

All actions share a purchase model and a purchaser-margin model. The margin
prediction is averaged with the training purchasers' mean for the requested
action. Non-purchasers must have zero target; purchaser losses remain negative.
"""
    columns = _features(num, cat)
    _exploration(data, target)
    if not data.purchase_14d.isin([0, 1]).all() or data.purchase_14d.nunique() != 2:
        raise ValueError('Hurdle training needs both binary purchase outcomes')
    bought = data.purchase_14d.eq(1)
    if not np.allclose(data.loc[~bought, target], 0):
        raise ValueError('Non-purchasers must have zero NCM for hurdle factorization')
    means = data.loc[bought].groupby('offer_type')[target].mean()
    if set(means.index) != set(ACTIONS):
        raise ValueError('Each action needs at least one purchaser')
    features = data[columns].assign(_action=data.offer_type.to_numpy())
    purchase = _hurdle_pipeline(num, cat, purchase=True).fit(features, data.purchase_14d)
    margin = _hurdle_pipeline(num, cat).fit(features.loc[bought], data.loc[bought, target])
    return {action: HurdleActionModel(purchase, margin, action, float(means[action]))
            for action in ACTIONS}


def predict_actions(models, features):
    predictions = np.column_stack([models[a].predict(features) for a in ACTIONS]).astype(float)
    if predictions.shape != (len(features), len(ACTIONS)) or not np.isfinite(predictions).all():
        raise ValueError('Action predictions must be finite and cover all four actions')
    return predictions


@dataclass
class ConstantPredictor:
    mean: float

    def predict(self, features):
        return np.full(len(features), self.mean, dtype=float)


def _reference_xgb(num, cat):
    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median', add_indicator=True), num),
        ('cat', _categorical(), cat),
    ])
    learner = XGBRegressor(objective='reg:squarederror', n_estimators=100,
        max_depth=1, learning_rate=.03, min_child_weight=10, subsample=.8,
        colsample_bytree=.8, reg_lambda=5., tree_method='hist', random_state=42, n_jobs=4)
    return Pipeline([('pre', pre), ('xgb', learner)])


def fit_nuisance_models(train, num, cat):
    """Fit per-action outcome regressions for historical doubly robust evaluation.

These are the fixed reference regressions used in the preceding CP07 audit;
they are not candidates for the deployed profit model.
"""
    columns = _features(num, cat)
    _exploration(train, 'ncm_14d')
    return {action: _reference_xgb(num, cat).fit(
        train.loc[train.offer_type.eq(action), columns],
        train.loc[train.offer_type.eq(action), 'ncm_14d']) for action in ACTIONS}


def fit_cost_models(train, num, cat, selection_campaign=None):
    """Choose per-offer XGBoost versus a mean using an earlier validation campaign.

Selection trains only on rows sent earlier whose outcomes have matured by the
validation send time; final fitting uses all supplied time-eligible rows. Set
CP06 explicitly for the final July fit so CP07 cannot select the cost-model
family. A tie favors the simpler action mean. Campaign names do not imply time.
"""
    columns = _features(num, cat)
    _exploration(train, 'promo_cost_14d')
    if train.promo_cost_14d.lt(0).any():
        raise ValueError('Promotion-cost targets must be nonnegative')
    required = {'campaign_id', 'send_ts', 'outcome_end_ts'}
    if not required.issubset(train):
        raise ValueError('Cost selection requires campaign_id, send_ts and outcome_end_ts')
    train = train.copy()
    for column in ('send_ts', 'outcome_end_ts'):
        train[column] = pd.to_datetime(train[column], errors='raise', format='mixed')
    if train[list(required)].isna().any().any() or train.outcome_end_ts.le(train.send_ts).any():
        raise ValueError('Cost selection needs valid send times and later outcome ends')
    if not train.groupby('campaign_id').send_ts.nunique().eq(1).all():
        raise ValueError('Each cost-selection campaign must have one send timestamp')
    campaign_times = train.groupby('campaign_id').send_ts.first()
    validation = selection_campaign if selection_campaign is not None else campaign_times.idxmax()
    if validation not in set(train.campaign_id):
        raise ValueError('Cost selection campaign must be in the training period')
    validation_cutoff = campaign_times.loc[validation]
    result = {}
    for action in ACTIONS[1:]:
        rows = train[train.offer_type.eq(action)]
        earlier = rows[rows.send_ts.lt(validation_cutoff) & rows.outcome_end_ts.le(validation_cutoff)]
        valid = rows[rows.campaign_id.eq(validation)]
        if earlier.empty or valid.empty:
            raise ValueError('Cost selection needs earlier training rows and validation support')
        candidate = _reference_xgb(num, cat).fit(earlier[columns], earlier.promo_cost_14d)
        prediction = np.maximum(candidate.predict(valid[columns]), 0)
        xgb_rmse = float(np.sqrt(np.mean((valid.promo_cost_14d - prediction) ** 2)))
        mean_rmse = float(np.sqrt(np.mean((valid.promo_cost_14d - earlier.promo_cost_14d.mean()) ** 2)))
        use_xgb = xgb_rmse < mean_rmse
        fitted = (_reference_xgb(num, cat).fit(rows[columns], rows.promo_cost_14d)
                  if use_xgb else ConstantPredictor(float(rows.promo_cost_14d.mean())))
        fitted.selection_ = dict(model='xgb' if use_xgb else 'action_mean',
            validation_campaign=validation, xgb_rmse=xgb_rmse, mean_rmse=mean_rmse,
            selection_training_rows=len(earlier), validation_rows=len(valid), final_training_rows=len(rows))
        result[action] = fitted
    return result


def predict_costs(models, features):
    predictions = np.column_stack([np.zeros(len(features))] +
        [np.maximum(models[a].predict(features), 0) for a in ACTIONS[1:]])
    if not np.isfinite(predictions).all():
        raise ValueError('Promotion-cost predictions must be finite')
    return predictions
