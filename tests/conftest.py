"""Small, internally reconciled extracts; no project dataset is required."""
import pandas as pd
import pytest

from promopilot.data import SCHEMA, TIMESTAMPS


@pytest.fixture
def raw_tables():
    rows = {
        "campaigns": [
            ["CP01", "2025-01-01 10:00", "First", "email", 14, "historical"],
            ["CP02", "2025-02-01 10:00", "Second", "email", 14, "historical"],
            ["CP08_POLICY", "2025-03-01 10:00", "Policy", "email", 14, "future"],
        ],
        "customers": [
            [1, "2024-01-01", "West", "search", "Standard", 0, "Home", "A", 1],
            [2, "2024-02-01", "East", None, "Standard", 1, None, "B", 1],
        ],
        "historical_assignments": [
            [11, "CP01", 1, "2025-01-01 10:00", "exploration", "PCT_10", 1],
            [12, "CP02", 1, "2025-02-01 10:00", "exploration", "FREE_SHIP", 1],
        ],
        "web_sessions": [
            [21, 1, "2024-12-30 12:00", "mobile", 3, 1, "Home", "search"],
            [22, None, "2025-01-01 12:00", "desktop", 1, 0, "Home", "direct"],
        ],
        "email_events": [[31, 11, 1, "CP01", "2025-01-01 11:00", "open"]],
        "orders": [
            [41, 1, "2025-01-03 10:00", "CP01", "campaign_attributed", "completed", 1, "PCT_10", 100., 10., 0., 90., 60., 4.],
            [42, 1, "2025-02-03 10:00", "CP02", "campaign_attributed", "completed", 1, "FREE_SHIP", 50., 0., 0., 50., 30., 4.],
            [43, 1, "2024-12-01 10:00", None, "organic_preperiod", "completed", 0, None, 40., 0., 5.99, 45.99, 20., 4.],
            [44, 1, "2025-01-05 10:00", "CP01", "campaign_attributed", "cancelled", 0, None, 0., 0., 0., 0., 0., 0.],
        ],
        "order_items": [[41, 1, 51, 2, 50., 30.], [42, 1, 51, 1, 50., 30.], [43, 1, 52, 1, 40., 20.]],
        "products": [[51, "Home", 50., 30., .1, 2.], [52, "Home", 40., 20., .1, 1.]],
        "returns": [[61, 41, "2025-01-08 10:00", 10., "partial"], [62, 41, "2025-01-20 10:00", 5., "partial"]],
        "offer_catalog": [[a, a, "documented"] for a in ["NO_OFFER", "PCT_10", "PCT_20", "FREE_SHIP"]],
        "policy_scoring_population": [[2, "2025-02-20 10:00", "CP08_POLICY"]],
    }
    result = {name: pd.DataFrame(values, columns=SCHEMA[name].split()) for name, values in rows.items()}
    for name, column in TIMESTAMPS.items():
        result[name][column] = pd.to_datetime(result[name][column], format="mixed")
    return result
