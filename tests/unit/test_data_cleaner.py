import pytest
from unittest.mock import patch, Mock
import os
import requests
import pandas as pd
from src.pipeline.data_cleaner import DataCleaner
from test_data.property_data_raw import property_data_raw
from test_data.property_data_cleaned import property_data_cleaned, property_data_flattened
from test_data.property_data_features import property_data_features


@pytest.fixture
def data_cleaner():
    return DataCleaner()
