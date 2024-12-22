import os
import requests
import pickle
from io import BytesIO

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from room_labeler_cnn import RoomClassifier, room_label_map
from room_scoring_cnn import RoomScore


LABEL_MODEL_PATH = "cnn_models/room_classifier.pth"
SCORING_MODEL_PATH = "cnn_models/room_scorer.pth"
LINEAR_REGRESSION_MODEL_PATH = "cnn_models/lm_with_images.sav"

ROOM_WEIGHTS = {
    "Bathroom": 1.8,
    "Kitchen": 2.0,
    "Living Room": 1.3,
    "Dining Room": 1.0,
    "Bedroom": 1.2,
    "Office": 0.5,
    "Closet": 0.4,
    "Basement": 0.6,
    "Stairs": 0.2,
    "Exterior": 1.0,
}


# Example Address: 1161 Natchez Dr College Station Texas 77845 --> streetAddress city state zip
def get_property_data(address) -> dict:
    url = "https://zillow56.p.rapidapi.com/search_address"
    querystring = {"address": address}

    headers = {"x-rapidapi-key": os.getenv("RAPID_API_KEY"), "x-rapidapi-host": os.getenv("RAPID_API_HOST")}

    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        property_data = _extract_property_data(response.json())
        return property_data
    else:
        print(
            f"Error fetching data for {
            address}: HTTP Status Code {response.status_code}"
        )
        return None


def get_images(property_data):
    address = property_data.get("address")
    image_urls = property_data.get("hugePhotos")
    if not address or not image_urls:
        return {}

    images = {}

    for i, image_data in enumerate(image_urls):
        url = image_data.get("url")
        if url:
            images[f"pic{i+1}_{address.replace(' ', '_')}"] = url  # Save URL instead of appending

    return images


def calculate_property_score(images, local=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_model = _load_model(LABEL_MODEL_PATH, len(room_label_map), device, model_type="classification")
    scoring_model = _load_model(SCORING_MODEL_PATH, 1, device, model_type="regression")

    label_map_reverse = {v: k for k, v in room_label_map.items()}

    room_scores = {room: [] for room in ROOM_WEIGHTS}

    for name, url in images.items():
        if not local:
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))

        else:
            img = Image.open(url)

        room_type = _predict_image(img, label_model, device, label_map_reverse, model_type="classification")

        if room_type in ROOM_WEIGHTS:
            score = _predict_image(img, scoring_model, device, model_type="regression")
            # Normalizes property score between 0-1
            normalized_score = (score - 1) / (9 - 1)
            room_scores[room_type].append(normalized_score)

    # Calculate weighted sum using ROOM_WEIGHTS
    total_weight = 0
    weighted_sum = 0
    for room, scores in room_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            weight = ROOM_WEIGHTS[room]
            weighted_sum += avg_score * weight
            total_weight += weight

    quality_score = weighted_sum / total_weight if total_weight > 0 else 0

    return quality_score


def calculate_property_price(property_data, images) -> float:
    try:
        lm = pickle.load(open(LINEAR_REGRESSION_MODEL_PATH, "rb"))
        quality_score = calculate_property_score(images)

        # Define the required columns in the correct order
        required_cols = [
            "zestimate",
            "bathrooms",
            "bedrooms",
            "livingArea",
            "lotSize",
            "yearBuilt",
            "taxAssessedValue",
            "quality_score",
        ]

        # Create a dictionary with the required data
        data_dict = {
            "zestimate": float(property_data.get("zestimate", 0)),
            "bathrooms": float(property_data.get("bathrooms", 0)),
            "bedrooms": float(property_data.get("bedrooms", 0)),
            "livingArea": float(property_data.get("livingArea", 0)),
            "lotSize": float(property_data.get("lotSize", 0)),
            "yearBuilt": float(property_data.get("yearBuilt", 0)),
            "taxAssessedValue": float(property_data.get("taxAssessedValue", 0)),
            "quality_score": quality_score,
        }

        # Convert to DataFrame with a single row
        df = pd.DataFrame([data_dict])

        # Ensure all columns are present and in the correct order
        df = df[required_cols]

        # Handle missing values
        df = df.fillna(0)

        # Make prediction
        prediction = lm.predict(df)[0]  # Get first prediction since we only have one row
        zestimate = property_data.get("zestimate", 0)

        return (float(prediction), float(zestimate))

    except Exception as e:
        print(f"Error in calculate_property_price: {str(e)}")
        raise


def _preprocess_image(image: Image):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def _load_model(model_path: str, num_classes: int, device, model_type="classification"):
    # Validate input parameters
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"Invalid number of classes: {num_classes}. Must be a positive integer.")

    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist at {model_path}")

    # Attempt to load model state dict
    if model_type == "classification":
        model = RoomClassifier(num_classes=num_classes)
    elif model_type == "regression":
        model = RoomScore(num_classes=1)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except (IOError, RuntimeError) as load_error:
        raise RuntimeError(f"Could not load model state dict: {load_error}")

    # Validate state dict
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as state_dict_error:
        raise ValueError(f"Model state dict is incompatible: {state_dict_error}")

    # Move model to device and set evaluation mode
    model.to(device)
    model.eval()

    return model


def _predict_image(image, model, device, label_map_reverse=None, model_type="classification"):
    image_tensor = _preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        if model_type == "classification":
            _, predicted = torch.max(outputs, 1)
            return label_map_reverse.get(predicted.item(), "Unknown")
        elif model_type == "regression":
            return outputs.squeeze().item()


def _extract_property_data(property_data):
    property_dict = {}
    property_dict["address"] = property_data.get("abbreviatedAddress")
    property_dict["city"] = property_data.get("city")
    property_dict["state"] = property_data.get("state")
    property_dict["zipcode"] = property_data.get("zipcode")
    property_dict["county"] = property_data.get("county")
    property_dict["countyFIPS"] = property_data.get("countyFIPS")
    property_dict["latitude"] = property_data.get("latitude")
    property_dict["longitude"] = property_data.get("longitude")
    property_dict["bathrooms"] = property_data.get("bathrooms")
    property_dict["bathroomsFull"] = property_data.get("bathroomsFull")
    property_dict["bathroomsHalf"] = property_data.get("bathroomsHalf")
    property_dict["bedrooms"] = property_data.get("bedrooms")
    property_dict["homeStatus"] = property_data.get("homeStatus")
    property_dict["homeType"] = property_data.get("homeType")
    property_dict["livingArea"] = property_data.get("livingArea")
    property_dict["lotSize"] = property_data.get("lotSize")
    property_dict["price"] = property_data.get("price")
    property_dict["currency"] = property_data.get("currency")
    property_dict["description"] = property_data.get("description")
    property_dict["contingentListingType"] = property_data.get("contingentListingType")
    property_dict["datePostedString"] = property_data.get("datePostedString")
    property_dict["datePriceChanged"] = property_data.get("datePriceChanged")
    property_dict["dateSoldString"] = property_data.get("dateSoldString")
    property_dict["daysOnZillow"] = property_data.get("daysOnZillow")
    property_dict["favoriteCount"] = property_data.get("favoriteCount")
    property_dict["isListedByOwner"] = property_data.get("isListedByOwner")
    property_dict["isNonOwnerOccupied"] = property_data.get("isNonOwnerOccupied")
    property_dict["isPreforeclosureAuction"] = property_data.get("isPreforeclosureAuction")
    property_dict["isBankOwned"] = property_data.get("isBankOwned")
    property_dict["isForeclosure"] = property_data.get("isForeclosure")
    property_dict["isFSBA"] = property_data.get("isFSBA")
    property_dict["isFSBO"] = property_data.get("isFSBO")
    property_dict["isComingSoon"] = property_data.get("isComingSoon")
    property_dict["isForAuction"] = property_data.get("isForAuction")
    property_dict["isNewHome"] = property_data.get("isNewHome")
    property_dict["isOpenHouse"] = property_data.get("isOpenHouse")
    property_dict["isPending"] = property_data.get("isPending")
    property_dict["isZillowOwned"] = property_data.get("isZillowOwned")
    property_dict["keystoneHomeStatus"] = property_data.get("keystoneHomeStatus")
    property_dict["lastSoldPrice"] = property_data.get("lastSoldPrice")
    property_dict["listingTypeDimension"] = property_data.get("listingTypeDimension")
    property_dict["listing_agent"] = property_data.get("listing_agent")
    property_dict["livingAreaUnits"] = property_data.get("livingAreaUnits")
    property_dict["livingAreaUnitsShort"] = property_data.get("livingAreaUnitsShort")
    property_dict["livingAreaValue"] = property_data.get("livingAreaValue")
    property_dict["lotAreaUnits"] = property_data.get("lotAreaUnits")
    property_dict["lotAreaValue"] = property_data.get("lotAreaValue")
    property_dict["taxAssessedValue"] = property_data.get("taxAssessedValue")
    property_dict["taxAssessedYear"] = property_data.get("taxAssessedYear")
    property_dict["taxHistory"] = property_data.get("taxHistory")
    property_dict["appliances"] = property_data.get("appliances")
    property_dict["architecturalStyle"] = property_data.get("architecturalStyle")
    property_dict["heating"] = property_data.get("heating")
    property_dict["cooling"] = property_data.get("cooling")
    property_dict["fencing"] = property_data.get("fencing")
    property_dict["flooring"] = property_data.get("flooring")
    property_dict["basement"] = property_data.get("basement")
    property_dict["yearBuilt"] = property_data.get("yearBuilt")
    property_dict["laundryFeatures"] = property_data.get("laundryFeatures")
    property_dict["levels"] = property_data.get("levels")
    property_dict["stories"] = property_data.get("stories")
    property_dict["parkingFeatures"] = property_data.get("parkingFeatures")
    property_dict["structureType"] = property_data.get("structureType")
    property_dict["thumb_url"] = property_data.get("thumb_url")
    property_dict["tourEligibility"] = property_data.get("tourEligibility")
    property_dict["tourPhotos"] = property_data.get("tourPhotos")
    property_dict["timeOnZillow"] = property_data.get("timeOnZillow")
    property_dict["timeZone"] = property_data.get("timeZone")
    property_dict["virtualTourUrl"] = property_data.get("virtualTourUrl")
    property_dict["zipcodeSearchUrl"] = property_data.get("zipcodeSearchUrl")
    property_dict["zestimate"] = property_data.get("zestimate")
    property_dict["rentZestimate"] = property_data.get("rentZestimate")
    property_dict["hugePhotos"] = property_data.get("hugePhotos", [])
    property_dict["agentEmail"] = property_data.get("agentEmail")
    property_dict["agentLicenseNumber"] = property_data.get("agentLicenseNumber")
    property_dict["agentName"] = property_data.get("agentName")
    property_dict["agentPhoneNumber"] = property_data.get("agentPhoneNumber")
    property_dict["attributionTitle"] = property_data.get("attributionTitle")
    property_dict["brokerName"] = property_data.get("brokerName")
    property_dict["brokerPhoneNumber"] = property_data.get("brokerPhoneNumber")
    property_dict["buyerAgentName"] = property_data.get("buyerAgentName")
    property_dict["buyerBrokerageName"] = property_data.get("buyerBrokerageName")
    return property_dict


# Example Usage
def main():
    address = "1161 Natchez Dr College Station Texas 77845"
    quality_score = calculate_property_score(address)
    print(f"Property Quality Score for {address}: {quality_score}")


if __name__ == "__main__":
    main()
