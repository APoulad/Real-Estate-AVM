import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm.auto import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


class ScoreDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Pre-load and pre-process all images
        logger.info("Starting image pre-loading...")
        self.image_cache = self._preload_images()
        logger.info(f"Loaded {len(self.image_cache)} images")

    def _preload_images(self):
        from PIL import Image

        image_cache = []
        skipped_images = 0

        # Use tqdm for progress tracking during pre-loading
        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Preloading Images"):
            img_dir = os.path.join(self.root_dir, str(row["property_address"]))

            try:
                for img_filename in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, img_filename)

                    try:
                        # Open and transform image
                        img = Image.open(img_path).convert("RGB")
                        transformed_img = self.transform(img)

                        # Parse score
                        if isinstance(row["scores"], str):
                            scores = ast.literal_eval(row["scores"])
                            avg_score = float(np.mean(scores))
                        elif isinstance(row["scores"], (list, np.ndarray)):
                            avg_score = float(np.mean(row["scores"]))
                        else:
                            skipped_images += 1
                            continue

                        image_cache.append((transformed_img, avg_score))

                    except Exception as e:
                        logger.warning(f"Error processing {img_path}: {e}")
                        skipped_images += 1

            except FileNotFoundError:
                logger.warning(f"No images found for {row['property_address']}")
                skipped_images += 1

        logger.info(f"Skipped {skipped_images} problematic images")
        return image_cache

    def __len__(self):
        return len(self.image_cache)

    def __getitem__(self, idx):
        return self.image_cache[idx]


def create_mps_optimized_dataloader(data, root_dir, batch_size=64):
    """Create an optimized DataLoader for MPS."""
    dataset = ScoreDataset(data, root_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,  # Important for MPS
        num_workers=0,  # MPS works best with 0 workers
    )


class RoomScore(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Use a lighter backbone
        self.backbone = models.mobilenet_v2(weights="IMAGENET1K_V1")

        # Modify final layers for regression
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def train_mps_optimized(data, root_dir, num_epochs=10, batch_size=64, learning_rate=0.001):
    # Logging setup
    logger.info("Starting MPS Optimized Training")
    logger.info(f"Epochs: {num_epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    # Ensure MPS is available
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available")

    # Create DataLoader
    dataloader = create_mps_optimized_dataloader(data, root_dir, batch_size)

    # Initialize model on MPS
    device = torch.device("mps")
    model = RoomScore().to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with tqdm
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Progress bar for epoch
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", colour="green")

        for images, labels in progress_bar:
            # Move to MPS device
            images = images.to(device)
            labels = labels.float().to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        # Epoch summary
        avg_epoch_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")

    logger.info("Training completed successfully")
    return model


def main():
    from sklearn.model_selection import train_test_split

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    try:
        # Load data
        logger.info("Loading dataset...")
        data = pd.read_csv("scores_labeled_properties.csv")

        # Split data
        train_data, _ = train_test_split(data, test_size=0.2, random_state=42)

        # Train model
        logger.info("Starting model training...")
        model = train_mps_optimized(train_data, root_dir="data/raw/sample_property1", num_epochs=10, batch_size=64)

        # Save model
        model_path = "mps_optimized_model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
