import os
import pandas as pd
from quality_score import calculate_property_score

IMAGE_PATH = "data/raw/pictures"
CSV_PATH = "data/raw/cleaned_property_data.csv"

# Load the CSV
df = pd.read_csv(CSV_PATH)

# Add a column for quality_score if it doesn't exist
if "quality_score" not in df.columns:
    df["quality_score"] = None


# Function to calculate scores and update CSV incrementally
def calculate_scores_and_update_csv(df, image_path, csv_path):
    for index, row in df.iterrows():
        if pd.notna(row["quality_score"]):  # Skip if score already calculated
            continue

        property_address = row["address"].replace(" ", "_")
        directory = f"{image_path}/{property_address}"
        try:
            # Get image filenames from the directory
            if os.path.exists(directory):
                images = {
                    img: os.path.join(directory, img)
                    for img in os.listdir(directory)
                    if os.path.isfile(os.path.join(directory, img))
                }
                score = calculate_property_score(images, local=True)
                print(f"Score for {property_address}: {score}")
            else:
                raise FileNotFoundError(f"Directory {directory} does not exist")
        except Exception as e:
            print(f"Error processing {property_address}: {e}")
            score = None

        # Update the score in the DataFrame
        df.at[index, "quality_score"] = score

        # Save the DataFrame back to the CSV after each update
        df.to_csv(csv_path, index=False)


# Run the calculation and incremental saving
calculate_scores_and_update_csv(df, IMAGE_PATH, CSV_PATH)
