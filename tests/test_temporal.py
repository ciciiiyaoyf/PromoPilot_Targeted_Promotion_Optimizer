"""Decision-time leakage and transaction-coverage checks using synthetic rows."""
import pandas as pd
import pytest

from promopilot.coverage import build_transaction_coverage
from promopilot.features import build_features


@pytest.fixture
def history():
    cutoff = pd.Timestamp('2025-06-20 10:00:00')
    spine = pd.DataFrame(dict(assignment_id=['current'], customer_id=['c1'], send_ts=[cutoff]))
    tables = {
        'customers': pd.DataFrame(dict(customer_id=['c1'], signup_date=[pd.Timestamp('2024-01-01')],
            is_member=[0], marketing_consent=[1], region=['west'], acquisition_channel=['web'],
            customer_tier=['Standard'], preferred_category=['shoes'], shipping_zone=['z1'])),
        'orders': pd.DataFrame(dict(order_id=['recent', 'old'], customer_id=['c1', 'c1'],
            campaign_id=['CP01', 'CP01'], order_source=['campaign_attributed', 'organic_preperiod'],
            order_ts=[cutoff-pd.Timedelta(days=30), cutoff-pd.Timedelta(days=100)],
            is_completed=[True, True], net_revenue=[100., 80.], cogs=[40., 30.], shipping_cost=[5., 5.],
            merchandise_gross_revenue=[110., 80.], discount_amount=[10., 0.],
            shipping_revenue=[0., 0.], offer_redeemed=[1, 0])),
        'returns': pd.DataFrame(dict(order_id=['recent'], return_ts=[cutoff-pd.Timedelta(days=20)],
            refund_amount=[10.])),
        'historical_assignments': pd.DataFrame(dict(assignment_id=['past'], customer_id=['c1'],
            campaign_id=['CP01'], send_ts=[cutoff-pd.Timedelta(days=60)],
            offer_type=['PCT_10'], email_delivered=[1])),
        'web_sessions': pd.DataFrame(dict(customer_id=['c1'], session_ts=[cutoff-pd.Timedelta(days=1)],
            pages_viewed=[2], added_to_cart=[1], category_viewed=['shoes'])),
        'email_events': pd.DataFrame(dict(assignment_id=['past'], campaign_id=['CP01'], customer_id=['c1'],
            event_ts=[cutoff-pd.Timedelta(days=2)], event_type=['open'])),
        'order_items': pd.DataFrame(dict(order_id=['recent', 'old'], product_id=['p1', 'p1'], quantity=[2, 1])),
        'products': pd.DataFrame(dict(product_id=['p1'], category=['shoes'])),
    }
    return spine, tables


@pytest.mark.parametrize('inclusive', [False, True])
def test_future_events_cannot_change_any_predictor(history, inclusive):
    spine, tables = history
    before, _ = build_features(spine, tables, inclusive=inclusive)
    future = spine.send_ts.iloc[0]+pd.Timedelta(seconds=1)
    changed = {name: table.copy(deep=True) for name, table in tables.items()}
    for name, timestamp in {'orders':'order_ts', 'returns':'return_ts',
                            'historical_assignments':'send_ts', 'web_sessions':'session_ts',
                            'email_events':'event_ts'}.items():
        extra = changed[name].iloc[[0]].copy()
        extra[timestamp] = future
        if name == 'orders':
            extra['order_id'] = 'future_order'
        elif name == 'historical_assignments':
            extra['assignment_id'] = 'future_assignment'
        changed[name] = pd.concat([changed[name], extra], ignore_index=True)
    changed['order_items'] = pd.concat([changed['order_items'],
        changed['order_items'].iloc[[0]].assign(order_id='future_order', quantity=1000)], ignore_index=True)
    after, _ = build_features(spine, changed, inclusive=inclusive)
    pd.testing.assert_frame_equal(before, after)


def test_cutoff_and_lookback_boundaries(history):
    spine, tables = history
    cutoff = spine.send_ts.iloc[0]
    tables['orders'].loc[1, 'order_ts'] = cutoff-pd.Timedelta(days=90)
    tables['orders'] = pd.concat([tables['orders'],
        tables['orders'].iloc[[0]].assign(order_id='at_cutoff', order_ts=cutoff)], ignore_index=True)
    tables['web_sessions'].loc[0, 'session_ts'] = cutoff
    tables['email_events'].loc[0, 'event_ts'] = cutoff
    historical, audit = build_features(spine, tables)
    july, july_audit = build_features(spine, tables, inclusive=True)
    assert historical.loc[0, 'observed_orders_90d'] == 2
    assert historical.loc[0, 'observed_orders_30d'] == 1
    assert july.loc[0, 'observed_orders_90d'] == 3
    assert july.loc[0, 'observed_orders_30d'] == 2
    assert historical.loc[0, ['sessions_30d', 'prior_email_opens']].eq(0).all()
    assert july.loc[0, ['sessions_30d', 'prior_email_opens']].eq(1).all()
    assert all(lag < pd.Timedelta(0) for lag in audit.values())
    assert all(lag <= pd.Timedelta(0) for lag in july_audit.values())


def test_partial_refunds_do_not_multiply_orders_or_items(history):
    spine, tables = history
    cutoff = spine.send_ts.iloc[0]
    tables['returns'] = pd.DataFrame(dict(order_id=['recent']*3,
        return_ts=[cutoff-pd.Timedelta(days=20), cutoff-pd.Timedelta(days=10), cutoff],
        refund_amount=[10., 5., 7.]))
    historical, _ = build_features(spine, tables)
    july, _ = build_features(spine, tables, inclusive=True)
    assert historical.loc[0, 'observed_historical_ncm_90d'] == 40  # 100-40-5-15
    assert july.loc[0, 'observed_historical_ncm_90d'] == 33
    assert historical.loc[0, 'observed_units_purchased_90d'] == 2
    assert historical.loc[0, 'observed_orders_90d'] == 1


def test_current_campaign_is_not_counted_as_prior_exposure(history):
    spine, tables = history
    current = tables['historical_assignments'].iloc[[0]].assign(
        assignment_id='current', campaign_id='CP02', send_ts=spine.send_ts.iloc[0])
    tables['historical_assignments'] = pd.concat([tables['historical_assignments'], current], ignore_index=True)
    features, _ = build_features(spine, tables)
    assert features.loc[0, 'prior_emails_delivered'] == 1
    assert features.loc[0, 'prior_offer_redemption_rate'] == 1


def test_zero_recent_orders_stays_unknown_without_a_coverage_contract(history):
    spine, tables = history
    tables['orders']['order_ts'] = pd.Timestamp('2025-01-09')
    features, _ = build_features(spine, tables, inclusive=True)
    coverage = build_transaction_coverage(spine, tables['orders'], inclusive=True)
    assert features.loc[0, 'observed_orders_90d'] == 0
    assert features.loc[0, 'observed_prior_order_count'] == 2
    assert coverage.loc[0, 'observed_orders_90d'] == 0
    assert pd.isna(coverage.loc[0, 'complete_orders_90d'])
    assert coverage.loc[0, 'orders_90d_coverage_status'] == 'unverified'


def test_coverage_contract_must_be_available_and_cover_the_endpoint(history):
    spine, tables = history
    cutoff = spine.send_ts.iloc[0]
    contract = dict(scope='all_sources', start=cutoff-pd.Timedelta(days=90), end=cutoff,
                    available_as_of=cutoff+pd.Timedelta(seconds=1))
    manifest = {'coverage_intervals': [contract]}
    later = build_transaction_coverage(spine, tables['orders'], manifest=manifest)
    assert pd.isna(later.loc[0, 'complete_orders_90d'])
    contract['available_as_of'] = cutoff
    historical = build_transaction_coverage(spine, tables['orders'], manifest=manifest)
    july = build_transaction_coverage(spine, tables['orders'], manifest=manifest, inclusive=True)
    assert historical.loc[0, 'complete_orders_90d'] == 1
    assert pd.isna(july.loc[0, 'complete_orders_90d'])
    contract['end_inclusive'] = True
    july = build_transaction_coverage(spine, tables['orders'], manifest=manifest, inclusive=True)
    assert july.loc[0, 'complete_orders_90d'] == 1


def test_coverage_contracts_need_no_gap_and_respect_customer_scope(history):
    spine, tables = history
    cutoff = spine.send_ts.iloc[0]
    split = cutoff-pd.Timedelta(days=45)
    first = dict(scope='all_sources', start=cutoff-pd.Timedelta(days=90), end=split, available_as_of=cutoff)
    second = dict(scope='all_sources', start=split, end=cutoff, available_as_of=cutoff, customer_ids=['c1'])
    manifest = {'coverage_intervals': [second, first]}
    assert build_transaction_coverage(spine, tables['orders'], manifest=manifest).complete_orders_90d.notna().all()
    second['start'] = split+pd.Timedelta(seconds=1)
    assert build_transaction_coverage(spine, tables['orders'], manifest=manifest).complete_orders_90d.isna().all()
    second['start'] = split
    second['customer_ids'] = ['someone_else']
    assert build_transaction_coverage(spine, tables['orders'], manifest=manifest).complete_orders_90d.isna().all()


def test_future_customer_profile_is_rejected(history):
    spine, tables = history
    tables['customers']['signup_date'] = spine.send_ts.iloc[0]+pd.Timedelta(days=1)
    with pytest.raises(ValueError, match='signup'):
        build_features(spine, tables)
