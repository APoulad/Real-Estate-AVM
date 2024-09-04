from dotenv import load_dotenv
from typing import Dict, Any, Union, List
import pandas as pd
import ast
from src.utils import *

logger = setup_logger()


class DataProcessor:
    def __init__(self) -> None:
        pass


REFERENCE_CSV = "reference_properties.csv"
PROPERTY_COLLECTION = "columbus-sold"


def upload_images_to_bucket(city: str, address: str, image_urls: list):
    """Upload images to Firebase Storage bucket."""
    dir_name = address.replace(" ", "_").lower()
    city = city.lower()
    for i, image_data in enumerate(image_urls):
        url = image_data.get("url")
        if url:
            blob_name = f"images/{city}-sold/{dir_name}/pic{i+1}_{dir_name}.jpg"
            upload_image_to_gcs(os.getenv("FIREBASE_STORAGE_BUCKET"), blob_name, url)


def update_firestore(property_data: dict):
    """Update or add a property to Firestore"""
    coll = access_firestore_collection(PROPERTY_COLLECTION)
    doc_ref = coll.document(str(property_data["address"]))
    doc_ref.set(property_data, merge=True)
