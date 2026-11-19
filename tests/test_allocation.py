"""Check global allocation economics, not the solver's implementation details."""
from itertools import product
import numpy as np
import pandas as pd
import pytest

from promopilot.allocation import get_holdout, optimize_policy, assert_feasible, measurement_assignment
from promopilot.config import ACTIONS


def values():
    gains = [[0, 3, 14, -1], [0, -1, 12, 2], [0, 2, 8, 3], [0, 5, 7, 1], [0, 4, 10, 6]]
    return pd.DataFrame([dict(customer_id=i, action=a, predicted_ncm=10+gains[i][j],
        estimated_incremental_ncm=gains[i][j], estimated_promo_cost=[0, 2, 4, 1][j])
        for i in range(5) for j, a in enumerate(ACTIONS)])


def test_milp_matches_exhaustive_feasible_optimum():
    v = values()
    best = -np.inf
    for choices in product(range(4), repeat=5):
        if choices[0] != 0 or sum(a != 0 for a in choices) > 4:
            continue
        if choices.count(2) > .5 * sum(a != 0 for a in choices):
            continue
        selected = v.iloc[[4*i+a for i, a in enumerate(choices)]]
        if selected.estimated_promo_cost.sum() <= 9:
            best = max(best, selected.estimated_incremental_ncm.sum())
    policy = optimize_policy(v, budget=9, max_offer_share=.8, max_pct20_share=.5, holdout_ids=[0])
    assert policy.estimated_incremental_ncm.sum() == pytest.approx(best)
    assert policy.attrs['primary_mip_gap'] == 0
    reordered = optimize_policy(v.sample(frac=1, random_state=8), budget=9,
        max_offer_share=.8, max_pct20_share=.5, holdout_ids=[0])
    pd.testing.assert_frame_equal(policy, reordered)


def test_controls_reproducible_independent_of_row_order_and_rounded_up():
    ids = list(range(2776))
    control = get_holdout(ids)
    assert len(control) == 139
    np.testing.assert_array_equal(control, get_holdout(ids[::-1]))


def test_policy_no_offer_is_not_automatically_a_measurement_control():
    policy = optimize_policy(values(), budget=0, holdout_ids=[0])
    arms = measurement_assignment(policy)
    assert arms.recommended_action.eq('NO_OFFER').all()
    assert arms.randomized_arm.eq('CONTROL').sum() == 1
    assert arms.randomized_arm.eq('POLICY').sum() == 4


def test_malformed_values_and_infeasible_policy_rejected():
    with pytest.raises(ValueError, match='four actions'):
        optimize_policy(values().iloc[1:])
    policy = optimize_policy(values(), budget=0)
    policy.loc[0, 'recommended_action'] = 'PCT_20'
    with pytest.raises(ValueError, match='share'):
        assert_feasible(policy)


def test_negative_gain_support_offer_can_enable_profitable_pct20():
    v = values().loc[lambda x: x.customer_id.lt(2)].copy()
    v['estimated_incremental_ncm'] = [0, -1, 100, -2, 0, -1, -10, -2]
    v['predicted_ncm'] = v.estimated_incremental_ncm
    p = optimize_policy(v, budget=10, max_offer_share=1, max_pct20_share=.5)
    assert p.estimated_incremental_ncm.sum() == pytest.approx(99)
    assert set(p.recommended_action) == {'PCT_20', 'PCT_10'}
