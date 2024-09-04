import os
import pytest
from dotenv import load_dotenv
from src.pipeline.data_fetcher import DataFetcher

load_dotenv()


RAPID_API_KEY = os.getenv("RAPID_API_KEY")
RAPID_API_HOST = os.getenv("RAPID_API_HOST")


@pytest.fixture
def data_fetcher():
    return DataFetcher(api_key=RAPID_API_KEY, api_host=RAPID_API_HOST)


@pytest.mark.integration
def test_fetch_recent_properties_integration(data_fetcher):
    zipcode = "43201"
    result = data_fetcher.fetch_recent_properties(zipcode=zipcode)
    assert result is not None
    assert isinstance(result, dict)
    for address, zpid in result.items():
        assert isinstance(address, str)
        assert isinstance(zpid, int)


@pytest.mark.integration
def test_fetch_property_data_integration(data_fetcher):
    zpid = "33842768"  # Use a known valid zpid for testing
    result = data_fetcher.fetch_property_data(zpid=zpid)
    assert result is not None
    assert "address" in result
    assert isinstance(result["address"], dict)
