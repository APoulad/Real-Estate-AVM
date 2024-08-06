import tqdm
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from src.utils import setup_logger

logger = setup_logger()


class DataFetcher:
    def __init__(self, api_key: str, api_host: str):
        self.api_key = api_key
        self.api_host = api_host

    def fetch_recent_properties(
        self,
        city: str = None,
        state: str = None,
        zipcode: str = None,
        status: str = "recentlySold",
        lastSoldDays: str = "7",
    ) -> Optional[dict]:
        """Fetch recentlySold or forSale listings from any city and date range on Zillow"""
        if not isinstance(status, str) or status not in ["recentlySold", "forSale"]:
            logger.error("Status must be 'recentlySold' or 'forSale'.")
            return None
        valid_days = ["any", "1", "7", "14", "30", "90", "6m", "12m", "24m", "36m"]
        if lastSoldDays not in valid_days:
            logger.error(f"lastSoldDays must be one of {valid_days}.")
            return None

        # Build query string
        if city and state:
            location = f"{city}, {state}"
        else:
            location = zipcode

        url = "https://zillow56.p.rapidapi.com/search"
        querystring = {
            "location": location,
            "output": "csv",
            "status": status,
            "sortSelection": "saved",
            "doz": lastSoldDays,
        }

        headers = {"x-rapidapi-key": self.api_key, "x-rapidapi-host": self.api_host}

        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 200:
            return {
                f"{item['streetAddress']}, {item['city']}, {item['state']} {item['zipcode']}": item["zpid"]
                for item in response.json()["results"]
            }
        else:
            logger.error(f"Error fetching data for {location}: HTTP Status Code {response.status_code}")
            return None

    def fetch_property_data(self, zpid: str) -> Optional[Dict[str, Any]]:
        """Fetch property data from zpid on Zillow"""
        url = "https://zillow56.p.rapidapi.com/property"
        querystring = {"zpid": zpid}

        headers = {"x-rapidapi-key": self.api_key, "x-rapidapi-host": self.api_host}

        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for {zpid}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Error decoding JSON for {zpid}: {e}")
            return None

    def fetch_all_properties(self, zpids: List[str]) -> pd.DataFrame:
        """Fetch and process multiple properties"""
        all_properties = []
        for zpid in tqdm.tqdm(zpids, desc="Fetching properties"):
            data = self.fetch_property_data(zpid)
            if data is not None:
                all_properties.append(data)
            else:
                logger.warning(f"Failed to fetch property data for ZPID: {zpid}")

        return pd.DataFrame(all_properties)
