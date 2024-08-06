import pytest
from unittest.mock import patch, Mock
import os
import requests
import pandas as pd
from src.pipeline.data_fetcher import DataFetcher


@pytest.fixture
def data_fetcher():
    with patch.dict(os.environ, {"RAPID_API_KEY": "fake_api_key", "RAPID_API_HOST": "fake_api_host"}):
        fetcher = DataFetcher(api_key=os.getenv("RAPID_API_KEY"), api_host=os.getenv("RAPID_API_HOST"))
        yield fetcher


from unittest.mock import patch, Mock


@patch("src.pipeline.data_fetcher.requests.get")
def test_fetch_recent_properties_success(mock_get, data_fetcher):
    mock_response = Mock()
    mock_response.json.return_value = {
        "results": [
            {"streetAddress": "123 Main St", "city": "Columbus", "state": "OH", "zipcode": "43201", "zpid": 2134},
            {"streetAddress": "456 Elm St", "city": "Columbus", "state": "OH", "zipcode": "43201", "zpid": 5678},
        ]
    }
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    expected_result = {"123 Main St, Columbus, OH 43201": 2134, "456 Elm St, Columbus, OH 43201": 5678}

    result = data_fetcher.fetch_recent_properties(zipcode="43201")
    assert result == expected_result
    mock_get.assert_called_once()


@patch("src.pipeline.data_fetcher.requests.get")
def test_fetch_recent_properties_failure(mock_get, data_fetcher):
    mock_response = Mock()
    mock_response.status_code = 404
    mock_get.return_value = mock_response

    result = data_fetcher.fetch_recent_properties(zipcode="43201")
    assert result is None
    mock_get.assert_called_once()


@patch("src.pipeline.data_fetcher.requests.get")
def test_fetch_property_data_success(mock_get, data_fetcher):
    mock_response = Mock()
    expected_result = {"property": "details"}
    mock_response.json.return_value = expected_result
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    result = data_fetcher.fetch_property_data(zpid="12345")
    assert result == expected_result
    mock_get.assert_called_once_with(
        "https://zillow56.p.rapidapi.com/property",
        headers={"x-rapidapi-key": "fake_api_key", "x-rapidapi-host": "fake_api_host"},
        params={"zpid": "12345"},
    )


@patch("src.pipeline.data_fetcher.requests.get")
def test_fetch_property_data_failure(mock_get, data_fetcher):
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    mock_get.return_value = mock_response

    result = data_fetcher.fetch_property_data(zpid="12345")
    assert result is None
    mock_get.assert_called_once()


@patch.object(DataFetcher, "fetch_property_data")
def test_fetch_all_properties(mock_fetch_property_data, data_fetcher):
    mock_fetch_property_data.side_effect = [
        {"zpid": "12345", "details": "details1"},
        {"zpid": "67890", "details": "details2"},
    ]

    zpids = ["12345", "67890"]
    result = data_fetcher.fetch_all_properties(zpids)

    expected_df = pd.DataFrame([{"zpid": "12345", "details": "details1"}, {"zpid": "67890", "details": "details2"}])

    pd.testing.assert_frame_equal(result, expected_df)
    assert mock_fetch_property_data.call_count == len(zpids)
