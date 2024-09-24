# Detailed Work Plan for AVM

## Introduction

Real estate investors rely on accurate pricing predictions to make informed decisions about property investments. These predictions typically require analyzing structured data like property size, location, and comparable sales (comps), as well as visual data such as property images. Given the rapid pace of real estate markets, manually evaluating numerous properties can be time-consuming and inefficient. Our goal is to create an Automated Valuation Model (AVM) that uses advanced machine-learning techniques to streamline the valuation process. By integrating visual features from images with structured property data, the system will help investors make quicker, data-driven decisions and reduce missed opportunities.

The AVM will use a Convolutional Neural Network (CNN) to extract key visual features like property condition, style, and quality from images. These visual features will be combined with structured data to create a unified property embedding to represent the property. Our will then dynamically prioritize the most relevant comps for each property. After identifying a baseline price from these comps, the system will adjust for factors such as location, size, and unique features to provide a tailored price prediction.

This project involves several components: a hosted image labeler to label the data for our CNN, a multi-step AI model at the heart of our AVM, and a small UI for the demo of our AVM.

## 1st Bi-weekly milestone (Oct 14, 2024)

The objective for this milestone is to deploy the image labeling web application and ensure that it is fully operational, collecting labeled results, and tracking user sessions. The milestone will encompass the completion of the backend infrastructure, frontend UI/UX, and the deployment process through GCP and Vercel.

## Backend

The backend will be built using Flask, with multiple API endpoints to manage the image labeling process, user sessions, and results.

### API Endpoints

##### `get_image` (GET):

**Functionality**: Fetch a random property image from GCP Storage that hasn’t been labeled yet. Mark the image as "served" in Firestore to avoid duplicates.

**Implementation**:

- Query GCP Firestore for images where `served = False`.
- Retrieve a random image from GCP Storage using the `image_path` stored in Firestore.
- Set the `served` field to `True` to ensure no other users are served the same image.

**Notes**: Handle cases where no unlabeled images are available.

---

##### `start_session` (POST):

**Functionality**: Initialize a new user session with a username and session type (room labeling or score labeling).

**Implementation**:

- Store session information in Firestore, including `username`, `session type`, and the timestamp when the session starts.

**Notes**: The session data should also track which images have been labeled in the session based on the `image_path/id`.

---

##### `update_label` (POST):

**Functionality**: Update Firestore with the room type or score for a specific image, set the image’s labeled status to `True`, and increment the session score.

**Implementation**:

- Receive the labeled image ID, room type/score, and session information.
- Update the Firestore document associated with the image with the room type or score.
- Increment the user’s session score in Firestore.

**Room Types**:

- Exterior, Bathroom, Bedroom, Kitchen, Living Room, Basement, Attic, Garage, Other.

**Scores**: A scale from 1-9. (Descriptions for each score will be added later).

**Notes**: Ensure robust error handling and validation for input data.

---

##### `reset_image_served` (GET):

**Functionality**: Reset any images that have been served but not labeled after a defined time period (stale images).

**Implementation**:

- Query Firestore for images with `served = True` and `labeled = False`.
- Reset `served = False` for these images, allowing them to be re-served to new users.

**Scheduler**: This endpoint should run periodically using a cloud scheduler (e.g., Cloud Scheduler) to prevent stale images.

---

##### `get_leaderboard` (GET):

**Functionality**: Retrieve a leaderboard of user session scores, sorted by the number of labeled images.

**Implementation**:

- Query Firestore for all user sessions and their scores.
- Return a sorted list of users with the highest labeling scores.

---

##### `skip_image` (POST):

**Functionality**: Allows users to skip an image and reset its served status to `False`.

**Implementation**:

- Receive the image ID and session information.
- Update the Firestore document to reset `served = False`.

Here's your text converted to Markdown:

## Google Cloud Platform (GCP) Services

#### Cloud Storage:

Used to store the property images that users will label.

---

#### Firestore:

Stores the property data, labels, and session information. Tracks:

- Property labels (room_type, scores, etc.).
- User sessions, including the number of images labeled and the type of session.

---

#### Artifact Registry:

Stores the Docker container for the Flask application.

---

#### Cloud Run:

Hosts the Flask API using the Docker container for scalable and secure execution.

## Frontend (HTML/Tailwind CSS/JavaScript)

The frontend will be a simple, intuitive web interface for users to label property images.

### Image Display:

The central part of the screen will be occupied by the image being labeled.

### Labeling Buttons:

For room labeling, the buttons will include:

- Exterior, Bathroom, Bedroom, Kitchen, Living Room, Basement, Attic, Garage, Other.

For score labeling, users will see buttons from 1 to 9 representing quality scores.

### Submit Button:

- Submits the user’s label or score to the backend (`update_label` endpoint).
- Once submitted, a new image will be loaded.

### Vercel Hosting:

The frontend will be hosted on Vercel for continuous deployment and easy scalability.

## Dataset

- **Source**: Zillow API via RapidAPI.
- **Size**: Approx. 9,000 sold property images from Columbus, Ohio (August 2022 - 2024).
- **Data Processing**: The images are pre-processed and stored in GCP Storage. Each image is linked with metadata (location, property type) in Firestore.

## 2nd Bi-weekly milestone (Nov 6, 2024)

The objective of this milestone is to start transitioning from the data collection phase (image labeling) to the model development phase, where we build the initial machine learning models for property price prediction. We will leverage the labeled image data from the image labeling web app, begin setting up the Convolutional Neural Network (CNN) for image analysis, and use structured data to create baseline regression models for price prediction.

## Collected Results from Image Labeler

### Verify Data Integrity:

- Review and clean the data collected from the labeling web app in Firestore.
- Ensure that all images have room type labels and multiple scores as expected.
- **Firestore Query**: Fetch all documents where `labeled=True` and export the data into a structured format using Pandas for further analysis.

### Dataset for Model Training:

- A dataset containing room types, scores, and any other labels that have been collected from the labeling process.
- Ensure that this data is properly formatted for training the CNN and baseline regression models.

---

## Start Building the CNN for Image Labeling

### Convolutional Neural Network (CNN) Setup:

- Use PyTorch to develop the initial CNN for image analysis. The CNN will focus on identifying important features from the images, such as room condition, layout, and quality.

### Training Pipeline:

- Build the training pipeline using PyTorch, where labeled images are passed through the CNN to extract key features.

### Initial CNN Model:

- A functional CNN model built using PyTorch, with the ability to classify room types and analyze room conditions.
- Begin training the CNN using the labeled image data.

---

## Build Baseline Regression Models Using Metadata

### Regression Model Setup:

- Use structured property data (metadata such as property size, location, amenities) to build baseline regression models for price prediction.
- These models will act as the foundation for comparing the performance of the final integrated model (CNN + structured data).
- Use scikit-learn, Pandas, and NumPy to build linear regression models and other basic models (e.g., decision trees) to predict property prices.

### Exploratory Data Analysis (EDA):

- Use Jupyter Notebooks for EDA on the structured dataset.

#### Charts and Visualizations:

- Generate charts to display trends and correlations in the data, such as the relationship between property size and price, or location and price.

#### Collinear Variables:

- Identify collinear variables that could affect the regression model’s accuracy.

### Model Training:

- Train baseline regression models using the structured data.
- Evaluate model performance using metrics such as Mean Squared Error (MSE) and R-squared.

## 3rd Bi-weekly milestone (Nov 20, 2024)

The objective of this milestone is to finalize the CNN for image processing and integrating both the structured property data and image data into a unified model. The integration of different machine learning techniques will help create a robust property valuation model, combining visual and structured data inputs. The goal is to enhance the model’s accuracy by utilizing diverse methods, which may include decision trees, random forests, gradient boosting, and ensemble learning.

## CNN Completion

### Final Training and Fine-Tuning:

- The CNN should now be fully trained using the labeled images from the image labeling process. It will extract key features from property images, such as:
  - **Room quality** (1-9).
  - **Room type** (classified in earlier stages).
- Fine-tune the CNN to improve its ability to detect subtle visual differences that might impact property value (e.g., high-end kitchen finishes vs. basic installations).
- Calculate the **average score** of each property, where each room type has a different weight that affects the overall score.

---

## Integrating Structured and Visual Data

### Data Fusion:

- The primary goal of this milestone is to integrate structured property data (e.g., location, size, amenities) with the visual data extracted from the CNN. Combining these two types of data will improve the accuracy of the property price predictions.

### Feature Fusion:

- Combine the features extracted from the CNN (image data) with structured data features. This will be achieved either through simple concatenation techniques or attention mechanisms.

---

## Exploring ML Techniques:

Below are potential techniques to try. This will be determined by how much time we have available to test other methods.

- **Ensemble Learning**: Combine multiple models (e.g., random forest, gradient boosting, CNN) to generate a more robust prediction.
- **Decision Trees**: Use decision trees or random forests to model non-linear relationships between structured data and CNN outputs.
- **Gradient Boosting (XGBoost/LightGBM)**: Gradient boosting methods can be applied to both structured and visual features to create a highly predictive model.
- **Neural Networks**: Use multi-layer perceptrons (MLPs) to process the concatenated data (structured + image features) for property price prediction.

## 4th Bi-weekly milestone (Dec 4, 2024)

The objective for the fourth milestone is to fully integrate the property price prediction model with the frontend, enabling users to input a property address and get a predicted price and relevant images. The focus will also be on data analysis and refinement, ensuring the model's accuracy and dynamic comp adjustment for the final presentation.

Anything relating to the UI not completed during this milestone will be pushed to the following two weeks as it relates directly to the presentation.

## Data Analysis and Refinement

### Model Validation and Refinement:

- Conduct a thorough evaluation of the property price prediction model by analyzing its performance on the test set.
- Refine the model based on feedback from metrics such as Mean Squared Error (MSE), R-squared (R²), and accuracy in predicting prices.
- **Analyze Edge Cases**: Identify properties where the model performs poorly and investigate whether errors stem from poor image quality, incomplete data, or model limitations.
- **Hyperparameter Tuning**: Fine-tune hyperparameters (e.g., learning rate, number of layers) to improve model accuracy and ensure it generalizes well across the dataset.

### Comps Adjustment:

- Dynamically find comps from the property embeddings and adjust the final output for improved results over the current models.

---

## Frontend Development: React and Tailwind CSS for UI

### Frontend UI Development:

- Develop the final user interface using React and Tailwind CSS. The UI will allow users to input a property address, which will query the model for a price prediction.

### Image Display:

- Display the corresponding property images along with the predicted price. The UI should load images dynamically based on the selected property.

---

## Backend Integration:

### API Development:

- Build API endpoints in Flask to handle requests from the frontend, retrieve the predicted price from the model, and return the relevant property images.

### GCP Integration:

- Use Google Cloud Storage (GCS) to store and retrieve property images. Integrate Firestore to retrieve structured data (e.g., property size, location).

---

## Final Data Pipeline: From Address to Prediction

### End-to-End Pipeline:

When a user inputs a property address, the system should:

1. **Make API Request to Zillow API via Rapid API**: Retrieve relevant data (e.g., property size, number of rooms) and images.
2. **Pass the data to the model**: Run the structured data and associated images through the trained AI model.
3. **Display Report**: Give a detailed view of the property with the final price prediction and property details, as well as any comparable properties (comps).

## Tech Stack and Libraries:

- Backend: Flask (Python), Firestore (NoSQL database), GCP Storage, GCP Cloud Run, GCP Artifact Registry
- Frontend:
  - Image Labeler: HTML, Tailwind CSS, JavaScript
  - Final UI: React
- ML/Data Analysis: PyTorch, Scikit-learn, Pandas, Numpy, Matplotlib, Jupyter Notebooks
- Deployment: GCP (Google Cloud Platform), Vercel, Docker
