import os
import pandas as pd

# Define the mapping of filename prefixes to room types
ROOM_TYPE_MAP = {
    "bath": "Bathroom",
    "bed": "Bedroom",
    "din": "Dining Room",
    "kitchen": "Kitchen",
    "living": "Living Room",
    "house": "Exterior",
    "stairs": "Stairs",
    "closet": "Closet",
    "basement": "Basement",
    "simple_home_office": "Office",
}


def generate_room_type_csv(input_folder, output_csv):
    """
    Generate a CSV file with property_id and room_type from image filenames.

    Args:
    input_folder (str): Path to the folder containing images
    output_csv (str): Path where the output CSV will be saved
    """
    # List to store image metadata
    image_data = []

    # Iterate through image files in the folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # Extract room type prefix
            prefix = filename.split("_")[0].lower()

            # Validate and map room type
            if prefix in ROOM_TYPE_MAP:
                image_data.append({"property_id": filename, "room_type": ROOM_TYPE_MAP[prefix]})
            else:
                print(f"Warning: Unrecognized prefix for file {filename}")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(image_data)
    df.to_csv(output_csv, index=False)

    print(f"CSV generated: {output_csv}")
    print(f"Total images processed: {len(df)}")


# Example usage
input_folder = "data/raw/kaggle/house_data/"
output_csv = "kaggle_labels.csv"
generate_room_type_csv(input_folder, output_csv)
