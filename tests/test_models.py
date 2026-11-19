"""Behavior that matters for offer comparisons and portable model scoring."""
import numpy as np
import pandas as pd
import pytest

from promopilot.config import ACTIONS
from promopilot.models import (fit_cost_models, fit_logistic_baseline,
                              fit_profit_models, predict_actions, predict_costs)


def training():
    rows = []
    for month, campaign in enumerate(('CP01', 'CP02', 'CP03'), start=1):
        send = pd.Timestamp(year=2025, month=month, day=1)
        for a, action in enumerate(ACTIONS):
            for i in range(20):
                bought = int(i % 3 == 0)
                rows.append(dict(campaign_id=campaign, offer_type=action,
                    send_ts=send, outcome_end_ts=send + pd.Timedelta(days=14),
                    assignment_source='exploration', x=float(i), category=None if i % 5 == 0 else 'known',
                    purchase_14d=bought, ncm_14d=bought * (-10 - 2*a),
                    promo_cost_14d=0. if action == 'NO_OFFER' else bought * (1+a)))
    return pd.DataFrame(rows)


def test_signed_profit_survives_two_part_model_and_action_conditioning():
    data = training()
    models = fit_profit_models(data, ['x'], ['category'])
    prediction = predict_actions(models, data[['x', 'category']])
    assert prediction.shape == (len(data), 4)
    assert (prediction < 0).all()  # Purchaser losses must not be clipped/logged away.
    assert np.any(prediction[:, 0] != prediction[:, 3])


@pytest.mark.parametrize('component', ['logistic', 'profit', 'cost'])
def test_storage_null_representation_does_not_change_predictions(component):
    data = training()
    other = data.copy()
    other['category'] = other.category.fillna(np.nan)
    features = ['x', 'category']
    if component == 'logistic':
        model = fit_logistic_baseline(data, ['x'], ['category'])
        first = model.predict_proba(data[features])
        second = model.predict_proba(other[features])
    elif component == 'profit':
        model = fit_profit_models(data, ['x'], ['category'])
        first, second = [predict_actions(model, frame[features]) for frame in (data, other)]
    else:
        model = fit_cost_models(data, ['x'], ['category'], selection_campaign='CP02')
        first, second = [predict_costs(model, frame[features]) for frame in (data, other)]
        assert (first[:, 0] == 0).all()
        assert (first >= 0).all()
    np.testing.assert_array_equal(first, second)


def test_purchase_baseline_cannot_use_offer_or_outcome_as_predictor():
    data = training()
    with pytest.raises(ValueError, match='not actions or outcomes'):
        fit_logistic_baseline(data, ['x'], ['category', 'offer_type'])


def test_profit_rejects_targeted_training_and_nonbuyer_profit():
    data = training()
    data.loc[0, 'assignment_source'] = 'business_rule'
    with pytest.raises(ValueError, match='exploration'):
        fit_profit_models(data, ['x'], ['category'])
    data.loc[0, 'assignment_source'] = 'exploration'
    data.loc[data.purchase_14d.eq(0), 'ncm_14d'] = 1
    with pytest.raises(ValueError, match='Non-purchasers'):
        fit_profit_models(data, ['x'], ['category'])


def test_cost_family_selection_cannot_see_later_campaign_targets():
    data = training()
    changed = data.copy()
    changed.loc[changed.campaign_id.eq('CP03'), 'promo_cost_14d'] = 500.
    old = fit_cost_models(data, ['x'], ['category'], selection_campaign='CP02')
    new = fit_cost_models(changed, ['x'], ['category'], selection_campaign='CP02')
    for action in ACTIONS[1:]:
        assert old[action].selection_ == new[action].selection_
    # Future targets affect the later refit, which is separate from family selection.
    assert not np.allclose(predict_costs(old, data), predict_costs(new, data))


def test_cost_selection_requires_an_earlier_training_campaign():
    with pytest.raises(ValueError, match='earlier training'):
        fit_cost_models(training(), ['x'], ['category'], selection_campaign='CP01')


@pytest.mark.parametrize('selection', [None, 'CP02'])
def test_cost_selection_uses_dates_not_lexical_campaign_names(selection):
    original = training()
    names = {'CP01': 'Z_LAST_LEXICALLY', 'CP02': 'A_FIRST_LEXICALLY', 'CP03': 'M_MIDDLE'}
    relabeled = original.assign(campaign_id=original.campaign_id.map(names))
    first = fit_cost_models(original, ['x'], ['category'], selection)
    second = fit_cost_models(relabeled, ['x'], ['category'], names.get(selection))
    for action in ACTIONS[1:]:
        expected = first[action].selection_.copy()
        expected['validation_campaign'] = names[expected['validation_campaign']]
        assert second[action].selection_ == expected
    np.testing.assert_array_equal(predict_costs(first, original), predict_costs(second, original))


def test_cost_selection_excludes_unmatured_training_labels():
    data = training()
    data.loc[data.campaign_id.eq('CP01'), 'outcome_end_ts'] = pd.Timestamp('2025-02-02')
    with pytest.raises(ValueError, match='earlier training'):
        fit_cost_models(data, ['x'], ['category'], selection_campaign='CP02')
