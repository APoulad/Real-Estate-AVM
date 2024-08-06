import os
import yaml
import requests
import logging.config
from google.cloud import storage, firestore
from google.oauth2 import service_account
import pandas as pd
from dotenv import load_dotenv
from logging import Logger
from typing import Optional

load_dotenv()

FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")
FIRESTORE_SERVICE_ACCOUNT_FILE = os.getenv("FIRESTORE_SERVICE_ACCOUNT_FILE")

# Setup Google Cloud credentials
credentials = service_account.Credentials.from_service_account_file(FIRESTORE_SERVICE_ACCOUNT_FILE)
storage_client = storage.Client(credentials=credentials)
db = firestore.Client(credentials=credentials)


def setup_logger(config_path: str = "config.yaml", logger_name: Optional[str] = None) -> Logger:
    """Setup logging config for entire system."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        if "logging" in config:
            log_dir = os.path.dirname(config["logging"]["handlers"]["file"]["filename"])
            os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
            logging.config.dictConfig(config["logging"])
        else:
            raise KeyError("The key 'logging' is not found in the config file.")
    if logger_name:
        return logging.getLogger(logger_name)
    return logging.getLogger()


logger = setup_logger()


def download_blob_as_dataframe(source_blob_name: str) -> pd.DataFrame:
    """Downloads a blob from the bucket and loads it into a pandas DataFrame."""
    bucket = storage_client.bucket(FIREBASE_STORAGE_BUCKET)
    blob = bucket.blob(source_blob_name)
    try:
        content = blob.download_as_text()
        return pd.read_csv(pd.compat.StringIO(content))
    except Exception as e:
        logger.error(f"Error downloading {source_blob_name}: {str(e)}")
        return pd.DataFrame()


def upload_dataframe_to_blob(blob_name: str, df: pd.DataFrame):
    """Upload a DataFrame to Google Cloud Storage as a CSV."""
    bucket = storage_client.bucket(FIREBASE_STORAGE_BUCKET)
    blob = bucket.blob(blob_name)

    try:
        csv_content = df.to_csv(index=False)
        blob.upload_from_string(csv_content, content_type="text/csv")
        logger.info(f"Successfully uploaded {blob_name}")
    except Exception as e:
        logger.error(f"Error uploading {blob_name}: {str(e)}")


def upload_image_to_gcs(blob_name: str, image_url: str) -> None:
    """Upload a single image to Google Cloud Storage."""
    bucket = storage_client.bucket(FIREBASE_STORAGE_BUCKET)
    blob = bucket.blob(blob_name)

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        blob.upload_from_string(response.content, content_type="image/jpeg")
        logger.info(f"Uploaded {blob_name} to {FIREBASE_STORAGE_BUCKET}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image {image_url}: {e}")
    except Exception as e:
        logger.error(f"Error uploading image {image_url}: {e}")


def access_firestore_collection(collection_name: str):
    """Access a Firestore collection."""
    try:
        collection_ref = db.collection(collection_name)
        docs = collection_ref.stream()
        for doc in docs:
            logger.info(f"Document ID: {doc.id}, Data: {doc.to_dict()}")
        return collection_ref
    except Exception as e:
        logger.error(f"Error accessing Firestore collection {collection_name}: {e}")
        return None
