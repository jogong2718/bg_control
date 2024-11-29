#import pytest
from unittest.mock import MagicMock
from pathlib import Path
import pandas as pd
from pathlib import Path
import sys
import os


# Mocking external dependencies
sys.modules['dataset_operations'] = MagicMock()
sys.modules['dataset_cleaner'] = MagicMock()

# Importing from the project
from meal_identification.modeling.train import (
    ScaledLogitTransformer, GMMHMM, train_model_instance, 
    load_data, xy_split, process_labels, load_model, save_model
)
from meal_identification.config import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR
)

# Define a fixture for sample data
import pytest
import pandas as pd

@pytest.fixture
def sample_data():
    """
    Fixture to provide a sample dataframe loaded from a CSV file.
    """
    file_path = "0_meal_identification/meal_identification/data/interim/2024-11-15_500030__i5mins_d4hrs_c5g_l3hrs_n4.csv"
    return pd.read_csv(file_path)


@pytest.fixture
def mock_model_paths(mocker):
    """
    Mock the paths for the model and data directories to use actual paths from the project structure.
    """
    mocker.patch("meal_identification.config.MODELS_DIR", Path("models/GaussianHMM_model"))
    mocker.patch("meal_identification.config.INTERIM_DATA_DIR", Path("data/interim"))
    mocker.patch("meal_identification.config.PROCESSED_DATA_DIR", Path("data/processed"))


def test_load_data(mocker, mock_model_paths, sample_data):
    """
    Test the load_data function for loading data from a CSV file.
    """
    # Mock the load_data function to return sample_data
    mocker.patch("meal_identification.modeling.train.load_data", return_value=sample_data)

    # Test function call
    data_path = INTERIM_DATA_DIR / "test_data.csv"
    result = load_data(data_path)

    # Assertions
    assert result.shape[1] == 10, "The dataframe does not have the expected number of columns."
    assert "bgl" in result.columns, "'bgl' column is missing in the dataframe."


def test_xy_split(sample_data):
    """
    Test splitting data into features (X) and target (Y).
    """
    X, Y = xy_split(sample_data)
    assert X.shape == (3, 1), "Features (X) do not have the correct shape."
    assert Y.shape == (3, 1), "Target (Y) does not have the correct shape."


def test_process_labels(sample_data):
    """
    Test the label processing for binary classification.
    """
    _, Y = xy_split(sample_data)
    processed_Y = process_labels(Y)
    assert set(processed_Y["msg_type"]) == {0, 1}, "Labels are not correctly processed."


def test_save_model(mocker):
    """
    Test saving the model to the specified path.
    """
    # Mocking the model and save function
    mock_model = MagicMock()
    mocker.patch("meal_identification.modeling.train.save_model")

    # Call save_model
    model_path = MODELS_DIR / "test_model.pkl"
    save_model(mock_model, model_path)

    # Verify the function was called
    save_model.assert_called_once_with(mock_model, model_path)


def test_train_model_instance(mocker, mock_model_paths):
    """
    Test training the model using train_model_instance.
    """
    # Mock the train_model_instance function
    mocker.patch("meal_identification.modeling.train.train_model_instance", return_value=MagicMock())

    # Call train_model_instance
    data_path = INTERIM_DATA_DIR / "test_data.csv"
    model_path = MODELS_DIR / "test_model.pkl"
    model = train_model_instance(
        model="GMMHMM",
        data_path=data_path,
        model_path=model_path,
        transformer=ScaledLogitTransformer()
    )

    # Verify the return type and functionality
    assert model is not None, "Model training returned None."
    assert isinstance(model, MagicMock), "Model is not of the expected type."


def test_train_full_model_pipeline(mocker, mock_model_paths):
    """
    Test the full model training process, including saving the model.
    """
    # Mock the train_model_instance and save_model functions
    mock_train_model_instance = mocker.patch("meal_identification.modeling.train.train_model_instance", return_value=MagicMock())
    mock_save_model = mocker.patch("meal_identification.modeling.train.save_model")

    # Perform full model training
    data_path = INTERIM_DATA_DIR / "test_data.csv"
    model_path = MODELS_DIR / "test_model.pkl"
    model = train_model_instance(
        model="GMMHMM",
        data_path=data_path,
        model_path=model_path,
        transformer=ScaledLogitTransformer()
    )
    save_model(model, model_path)

    # Assertions
    mock_train_model_instance.assert_called_once()
    mock_save_model.assert_called_once_with(model, model_path)
    assert model is not None, "Full training pipeline failed to return a trained model."
