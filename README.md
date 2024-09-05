# Real Estate AVM

## Table of Contents

- [Initial Project Proposal](#initial-project-proposal)
- [API Documentation](#api-documentation)
- [System Architecture](#system-architecture)
- [Technologies](#technologies)

## Initial Project Proposal

#### Overview:

Real estate investors rely heavily on accurate pricing predictions when considering a property for a potential investment. In order to make these estimates, investors analyze various sources of data including structured property data, images, and comparable sales (comps) to provide a holistic view of the target investment. With hundreds to thousands of properties in any given market, manually identifying great investment becomes an inefficient process and results in many missed opportunities due to fast-moving markets. Our goal is to develop a more efficient Automated Valuation Model (AVM) that leverages advanced machine learning techniques to accurately predict property prices. By identifying and adjusting for the most relevant comps, our system will help investors save time and make better data-driven decisions.

Our AVM will integrate both visual and structured data using machine learning techniques. First, a Convolutional Neural Network (CNN) will extract visual features from property images (e.g., room types, condition, and unique features like kitchen islands or walk-in showers). These features will be combined with structured data (e.g., location, size, and amenities) to create a comprehensive representation of each property. Using this representation, the model will identify the most relevant comparable properties and adjust the estimated price based on differences such as location, size, and specific features.

#### Dataset:

- A custom dataset collected from the Zillow API via RapidAPI, consisting of approximately 8,000 properties sold from August 2022 to August 2024 in Columbus, Ohio.
- Future expansion plans include gathering similar datasets from other Ohio markets while focusing on recent data to ensure relevance in changing economic conditions.

#### AVM Development Process:

- Step 1: Data Pipeline and Preprocessing

  Set up a data pipeline to automatically collect data from the Zillow API. Perform preprocessing steps, including cleaning and normalizing property data, handling missing values, and feature engineering to extract relevant structured data (e.g., property type, age, location coordinates, and amenities).

- Step 2: Feature Extraction and Embedding Creation

  Utilize a CNN to extract visual features from property images, such as room types, condition, and unique features. Combine these with structured data (e.g., property size, location) to create a unified property embedding.

- Step 3: Comparable Property Identification

  Use the unified embedding to find the most relevant comparable properties in the dataset based on similarity scores.

- Step 4: Price Adjustment

  Adjust the baseline price by accounting for key differences between the target property and its comps (e.g., location, age, and specific property features).

- Step 5: Model Training and Performance Evaluation

  Train the AVM on the dataset, starting with a baseline regression model using only structured property data and iteratively incorporating more complex features, such as the CNN and other ML techniques. Measure performance using metrics like Mean Squared Error (MSE) and R-squared (R²).

#### Data Labeling:

To label the data, we will develop a custom process that identifies room types, assesses their condition, and highlights relevant features such as kitchen islands or walk-in showers. This will involve using manual annotation tools to tag and describe various elements in the property images. Semi-automated techniques, supported by pre-trained models for object detection and image recognition, will help streamline the labeling process while maintaining high accuracy through human verification.

## API Documentation

Coming Soon!

## System Architecture

Detailed information about the system architecture is available in the [System Architecture](docs/v1/system-architecture.md) file.

## Technologies

- Backend: Python, Flask, RESTful API - Google Cloud Platform (Cloud Storage, Firestore, AI Platform)
- Machine Learning: TensorFlow, Scikit-learn, PyTorch
- Data Processing: Pandas, NumPy, Google Cloud Functions
