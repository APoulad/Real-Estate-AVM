import torch
from torchvision import transforms
from PIL import Image
import logging
import sys
from labeler_cnn import RoomClassifier, label_map  # Import your model and label map

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define a function to preprocess the input image
def preprocess_image(image_path: str):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        image = Image.open(image_path).convert("RGB")  # Open and convert to RGB
        return transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        sys.exit(1)


# Load the model
def load_model(model_path: str, num_classes: int, device):
    try:
        model = RoomClassifier(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)


# Predict function
def predict(image_path: str, model, device):
    # Preprocess the image
    image = preprocess_image(image_path).to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get the class index with the highest score

    # Reverse the label map to find the class name
    label_map_reverse = {v: k for k, v in label_map.items()}
    predicted_label = label_map_reverse.get(predicted.item(), "Unknown")

    return predicted_label


if __name__ == "__main__":
    # Input image path
    image_path = "data/raw/sample_property2/17_N_Parkview_Ave/pic9_17_N_Parkview_Ave.jpg"

    # Device configuration
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Model and label map details
    model_path = "room_classifier.pth"  # Path to the saved model
    num_classes = len(label_map)

    # Load model
    model = load_model(model_path, num_classes, device)

    # Predict the room type
    logger.info(f"Processing image: {image_path}")
    predicted_label = predict(image_path, model, device)

    # Output the result
    print(f"Predicted room type: {predicted_label}")
