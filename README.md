# Columbus Real Estate Predictor

## Project Overview

The primary objective of this project is to predict housing prices in the Columbus, Ohio market. The system will provide a full market analysis, individual property reports, and price predictions through a web application. The application will use data fetched from the Zillow API, process and store the data, train and deploy machine learning models for price prediction, and provide various endpoints for interacting with the data.

## Table of Contents

- [Project Overview](#project-overview)
- [Scope](#scope)
- [API Documentation](#api-documentation)
- [System Architecture](#system-architecture)
- [Technologies](#technologies)

## Scope

1. Data Collection:
   - Fetching real estate data from Zillow API using RapidAPI.
   - Gathering additional property images and metadata.
2. Data Processing:
   - Cleaning and preprocessing the fetched data.
   - Storing property images and metadata in Google Cloud Storage and Firestore.
3. Machine Learning:
   - Training machine learning models to predict property prices.
   - Models include Linear Regression, Hedonic Regression, and Ensemble models.
   - Evaluating models and storing them for deployment.
4. Web Application:
   - Creating a React frontend for user interaction.
   - Developing a Flask backend to serve API endpoints for data access and predictions.
   - Providing a comprehensive market analysis and individual property reports.
5. Deployment:
   - Deploying the application and models on Google Cloud Platform.
   - Ensuring scalability and reliability of the system.

## API Documentation

Coming Soon!

## System Architecture

Detailed information about the system architecture is available in the [System Architecture](docs/v1/system-architecture.md) file.

## Technologies

- Backend: Python, Flask, RESTful API - Google Cloud Platform (Cloud Storage, Firestore, AI Platform)
- Machine Learning: TensorFlow, Scikit-learn, PyTorch
- Data Processing: Pandas, NumPy, Google Cloud Functions
