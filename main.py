from flask import Flask, request, jsonify, render_template
from quality_score import get_property_data, get_images, calculate_property_price
from typing import Dict, Any, Optional
import logging
import logging.handlers
import traceback
import time
import os
from datetime import datetime

app = Flask(__name__)

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")


# Configure logging
def setup_logger():
    logger = logging.getLogger("property_analyzer")
    logger.setLevel(logging.DEBUG)

    # Format for logging
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s")

    # File handler for all logs
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename="logs/property_analyzer.log", when="midnight", interval=1, backupCount=30
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Error file handler
    error_file_handler = logging.handlers.TimedRotatingFileHandler(
        filename="logs/error.log", when="midnight", interval=1, backupCount=30
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()

# Cache for property data
property_data_cache: Dict[str, Dict[str, Any]] = {}


class RequestTimer:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        logger.info(f"Request to {self.endpoint} took {duration:.2f} seconds")


def validate_address(address: Optional[str]) -> tuple[bool, Optional[str]]:
    """Validate the address input."""
    logger.debug(f"Validating address: {address}")

    if not address:
        logger.warning("Address validation failed: Address is required")
        return False, "Address is required"
    if not isinstance(address, str):
        logger.warning(f"Address validation failed: Invalid type {type(address)}")
        return False, "Address must be a string"
    if len(address.strip()) == 0:
        logger.warning("Address validation failed: Empty address")
        return False, "Address cannot be empty"

    logger.debug("Address validation successful")
    return True, None


@app.route("/")
def index():
    logger.info("Serving index page")
    return render_template("index.html")


@app.route("/get-property-data", methods=["POST"])
def fetch_property_data():
    with RequestTimer("/get-property-data"):
        try:
            logger.info("Received property data request")
            data = request.get_json()

            if not data:
                logger.warning("No JSON data provided in request")
                return jsonify({"error": "No JSON data provided"}), 400

            address = data.get("address")
            logger.info(f"Processing request for address: {address}")

            is_valid, error_message = validate_address(address)
            if not is_valid:
                return jsonify({"error": error_message}), 400

            if address in property_data_cache:
                logger.info(f"Cache hit for address: {address}")
                return jsonify(
                    {
                        "message": "Data retrieved from cache",
                        "property_data": property_data_cache[address],
                        "status": "success",
                    }
                )

            logger.info(f"Cache miss for address: {address}. Fetching from API")
            property_data = get_property_data(address)

            if not property_data:
                logger.warning(f"No property data found for address: {address}")
                return jsonify({"error": "No property data found for this address", "status": "error"}), 404

            # Store complete data in cache
            property_data_cache[address] = property_data
            logger.debug(f"Cached property data for address: {address}")

            logger.info(f"Successfully retrieved property data for: {address}")
            return jsonify({"property_data": property_data, "status": "success"})

        except Exception as e:
            error_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.error(
                f"Error ID: {error_id} - Error in fetch_property_data: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            return (
                jsonify(
                    {
                        "error": "Internal server error while fetching property data",
                        "error_id": error_id,
                        "status": "error",
                    }
                ),
                500,
            )


@app.route("/get-property-images", methods=["POST"])
def fetch_property_images():
    with RequestTimer("/get-property-images"):
        try:
            logger.info("Received property images request")
            data = request.get_json()

            if not data:
                logger.warning("No JSON data provided in request")
                return jsonify({"error": "No JSON data provided"}), 400

            address = data.get("address")
            logger.info(f"Processing images request for address: {address}")

            is_valid, error_message = validate_address(address)
            if not is_valid:
                return jsonify({"error": error_message}), 400

            if address not in property_data_cache:
                logger.warning(f"Property data not found in cache for address: {address}")
                return (
                    jsonify({"error": "Property data not found. Please fetch property data first", "status": "error"}),
                    404,
                )

            property_data = property_data_cache[address]
            images = get_images(property_data)

            if not images:
                logger.warning(f"No images found for address: {address}")
                return jsonify({"error": "No images found for this property", "status": "error"}), 404

            image_urls = list(images.values())
            logger.info(f"Successfully retrieved {len(image_urls)} images for: {address}")
            return jsonify({"images": image_urls, "count": len(image_urls), "status": "success"})

        except Exception as e:
            error_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.error(
                f"Error ID: {error_id} - Error in fetch_property_images: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            return (
                jsonify(
                    {
                        "error": "Internal server error while fetching property images",
                        "error_id": error_id,
                        "status": "error",
                    }
                ),
                500,
            )


@app.route("/get-price-prediction", methods=["POST"])
def fetch_price_prediction():
    with RequestTimer("/get-price-prediction"):
        try:
            logger.info("Received price prediction request")
            data = request.get_json()

            if not data:
                logger.warning("No JSON data provided in request")
                return jsonify({"error": "No JSON data provided"}), 400

            address = data.get("address")
            logger.info(f"Processing price prediction for address: {address}")

            is_valid, error_message = validate_address(address)
            if not is_valid:
                return jsonify({"error": error_message}), 400

            if address not in property_data_cache:
                logger.warning(f"Property data not found in cache for address: {address}")
                return (
                    jsonify({"error": "Property data not found. Please fetch property data first", "status": "error"}),
                    404,
                )

            property_data = property_data_cache[address]
            images = get_images(property_data)

            if not images:
                logger.warning(f"No images found for price prediction for address: {address}")
                return jsonify({"error": "No images found for price prediction", "status": "error"}), 404

            logger.info(f"Calculating price prediction for: {address}")
            predicted_price, zestimate = calculate_property_price(property_data, images)

            difference_percentage = round(((predicted_price - zestimate) / zestimate) * 100, 2)
            logger.info(
                f"Price prediction completed for {address}. Difference from Zestimate: {difference_percentage}%"
            )

            return jsonify(
                {
                    "price_prediction": float(predicted_price),
                    "zestimate": float(zestimate),
                    "difference_percentage": difference_percentage,
                    "status": "success",
                }
            )

        except Exception as e:
            error_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.error(
                f"Error ID: {error_id} - Error in fetch_price_prediction: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            return (
                jsonify(
                    {
                        "error": "Internal server error while calculating price prediction",
                        "error_id": error_id,
                        "status": "error",
                    }
                ),
                500,
            )


@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({"error": "Endpoint not found", "status": "error"}), 404


@app.errorhandler(500)
def internal_error(error):
    error_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.error(f"Error ID: {error_id} - 500 error: {str(error)}\nTraceback:\n{traceback.format_exc()}")
    return jsonify({"error": "Internal server error", "error_id": error_id, "status": "error"}), 500


if __name__ == "__main__":
    logger.info("Starting Property Analyzer application")
    app.run(debug=True)
