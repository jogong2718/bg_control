import pytest
import pandas as pd
from datetime import datetime, timedelta

@pytest.fixture
def sample_meal_df():
    """Create a sample DataFrame for testing"""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='h')
    data = {
        'msg_type': ['ANNOUNCE_MEAL', '', 'ANNOUNCE_MEAL', 'ANNOUNCE_MEAL', ''],
        'food_g': [50.0, 0.0, 30.0, 20.0, 0.0],
        'day_start_shift': [1, 1, 1, 2, 2]
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def meal_length():
    """Standard meal length for testing"""
    return pd.Timedelta(hours=2)

@pytest.fixture
def n_top_carb_meals():
    """Number of top carbohydrate meals to keep"""
    return 3

@pytest.fixture
def coerce_interval():
    return pd.Timedelta(minutes=5)

@pytest.fixture
def df_to_coerce():
    dates = pd.date_range(start='2024-01-01', periods=10, freq='3min')
    df = pd.DataFrame({
        'date': dates,
        'bgl':  list(range(10, 101, 10)),
        'msg_type': ['', 'ANNOUNCE_MEAL', '', '', 'ANNOUNCE_MEAL', '', '', '', '', ''],
        'food_g': [0, 50, 0, 0, 75, 0, 0, 0, 0, 0]
    })
    df.set_index('date', inplace=True)
    return df

def pytest_generate_tests(metafunc):
    '''
    Uses pytest hooks to generate tests for multiple values of min_carbs
    See https://docs.pytest.org/en/stable/how-to/parametrize.html
    - Just adds fixture for min_carbs to all functions in metafunc (in this case, all tests since conftest.py is in the root tests folder)
    '''
    min_carbs_values = [10, 20, 30]
    if "min_carbs" in metafunc.fixturenames:
        metafunc.parametrize("min_carbs", min_carbs_values)
