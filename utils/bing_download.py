import requests
import os

# Configuration
subscription_key = os.getenv("BING_SEARCH_KEY")  # Set your Bing API key as an environment variable
search_url = "https://api.bing.microsoft.com/v7.0/images/search"
search_terms = [""]  # List of search terms
output_base_dir = "image_dataset"  # Base directory to save all images
num_images_per_term = 200  # Number of images to download per term


def download_images_for_term(search_term, output_dir, num_images):
    """Download images for a single search term."""
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_term, "count": 50, "offset": 0}
    downloaded = 0

    os.makedirs(output_dir, exist_ok=True)  # Create output directory for the search term

    while downloaded < num_images:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()
        search_results = response.json()

        for img in search_results.get("value", []):
            try:
                img_url = img["contentUrl"]
                img_data = requests.get(img_url, timeout=10).content
                with open(os.path.join(output_dir, f"{search_term}_{downloaded}.jpg"), "wb") as img_file:
                    img_file.write(img_data)
                downloaded += 1
                if downloaded >= num_images:
                    break
            except Exception as e:
                print(f"Failed to download image from {img_url}: {e}")

        params["offset"] += 50  # Increment offset for next batch

    print(f"Downloaded {downloaded} images for '{search_term}' to {output_dir}")


def main():
    for term in search_terms:
        output_dir = os.path.join(output_base_dir, term)  # Create a subdirectory for each search term
        download_images_for_term(term, output_dir, num_images_per_term)


if __name__ == "__main__":
    if not subscription_key:
        print("Error: Please set your Bing API key as the 'BING_SEARCH_KEY' environment variable.")
    else:
        main()
