from dotenv import load_dotenv
from typing import Dict, Any, Union, List
import pandas as pd
import ast
from src.utils import setup_logger

load_dotenv()

logger = setup_logger()


class DataCleaner:
    def __init__(self) -> None:
        pass

    def extract_property_data(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        keys = [
            "abbreviatedAddress",
            "city",
            "state",
            "zipcode",
            "county",
            "latitude",
            "longitude",
            "parcelNumber",
            "zpid",
            "homeType",
            "homeStatus",
            "keystoneHomeStatus",
            "bedrooms",
            "bathrooms",
            "livingArea",
            "lotSize",
            "livingAreaUnits",
            "yearBuilt",
            "roofType",
            "structureType",
            "price",
            "currency",
            "zestimate",
            "rentZestimate",
            "taxAssessedValue",
            "taxAssessedYear",
            "propertyTaxRate",
            "taxAnnualAmount",
            "datePostedString",
            "dateSoldString",
            "isListedByOwner",
            "description",
            "timeZone",
            "listing_agent",
            "priceHistory",
            "resoFacts",
            "atAGlanceFacts",
            "schools",
            "homeInsights",
        ]

        nested_keys = [
            "resoFacts",
            "atAGlanceFacts",
            "schools",
            "homeInsights",
            "listing_agent",
            "priceHistory",
            "hugePhotos",
        ]

        property_dict = {key: property_data.get(key, None) for key in keys}
        property_dict["hugePhotos"] = property_data.get("hugePhotos", [])

        for nested_key in nested_keys:
            nested_data = property_data.get(nested_key, {})
            if isinstance(nested_data, dict):
                for sub_key, sub_value in nested_data.items():
                    property_dict[f"{nested_key}_{sub_key}"] = sub_value
            elif isinstance(nested_data, list):
                # Ensure consistent length for nested lists
                for i, item in enumerate(nested_data):
                    if isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            property_dict[f"{nested_key}_{i}_{sub_key}"] = sub_value
                    else:
                        property_dict[f"{nested_key}_{i}"] = item

        return property_dict

    def extract_listing_agent_info(self, agent_info):
        try:
            agent_dict = ast.literal_eval(agent_info)
            return pd.Series(
                [
                    agent_dict.get("business_name", ""),
                    agent_dict.get("display_name", ""),
                    agent_dict.get("phone", {}).get("areacode", "")
                    + "-"
                    + agent_dict.get("phone", {}).get("prefix", "")
                    + "-"
                    + agent_dict.get("phone", {}).get("number", ""),
                ]
            )
        except:
            return pd.Series(["", "", ""])

    def clean_property_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(data, dict):
            cleaned_data = [self.extract_property_data(data)]
        elif isinstance(data, list):
            cleaned_data = [self.extract_property_data(property_data) for property_data in data]
        return pd.DataFrame(cleaned_data)
