# Shrimp Farming Prediction

This project aims to predict various metrics of shrimp farming, including Survival Rate (SR), Average Body Weight (ABW), Biomass, and Revenue. The project leverages feature engineering, machine learning models, and a Flask API for model predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Data Evaluation](#data-evaluation)
- [SR and ADG Calculation](#sr-and-adg-calculation)
- [Model Predictions](#model-predictions)
- [Feature Importance and Recommendations](#feature-importance-and-recommendations)
- [Setup Instructions](#setup-instructions)
- [Running the Code](#running-the-code)
- [API Deployment](#api-deployment)

## Project Overview

This project provides a comprehensive analysis and predictive modeling approach to optimize shrimp farming practices. By leveraging the provided datasets, we aim to deliver actionable insights and recommendations for achieving optimal results in shrimp farming.


## Project Structure

```plaintext
Shrimp-Farming-Prediction/
├── data/
│   ├── cycles.csv
│   ├── feeds.csv
│   ├── samplings.csv
│   ├── measurements.csv
│   ├── features_df.csv
│   ├── predictions.csv
├── models/
│   ├── rf_sr_model.pkl
│   ├── rf_abw_model.pkl
│   ├── xgb_biomass_model.pkl
│   ├── xgb_revenue_model.pkl
├── notebooks/
│   ├── JALA_Assessment.ipynb
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
├── api/
│   ├── app.py
├── README.md
├── requirements.txt


## Data Evaluation

We evaluated the completeness of each dataset, identifying missing values and potential inconsistencies. Below is a summary of the data evaluation:

### Cycles Dataset
- Non-null Count: 2617 entries
- Missing Values: species_id (814), finished_at (1), remark (1281), etc.

### Farms Dataset
- Non-null Count: 551 entries
- Missing Values: province (72), regency (93)

### Fasting Dataset
- Non-null Count: 38108 entries
- Missing Values: fasting (55)

### Feed Tray Dataset
- Non-null Count: 186664 entries
- Missing Values: remark (140283)

### Feeds Dataset
- Non-null Count: 706908 entries
- Missing Values: logged_at (21), quantity (21)

### Harvests Dataset
- Non-null Count: 8087 entries
- Missing Values: status (263), selling_price (1793)

### Measurements Dataset
- Non-null Count: 139050 entries
- Missing Values: multiple columns with varying missing counts

### Mortalities Dataset
- Non-null Count: 13221 entries
- Missing Values: None

### Ponds Dataset
- Non-null Count: 338 entries
- Missing Values: length (23), width (29), deep (92), max_seed_density (203)

### Samplings Dataset
- Non-null Count: 15032 entries
- Missing Values: remark (13693)

## Model Predictions

The following models were trained and deployed:

- **Survival Rate Prediction**: Random Forest
- **Average Body Weight Prediction**: Random Forest
- **Biomass Prediction**: XGBoost
- **Revenue Prediction**: XGBoost

## Feature Importance and Recommendations

### Feature Importance
- The most important features for each model were identified and analyzed. These features provide insights into the factors that significantly impact the predictions of SR, ABW, Biomass, and Revenue.

### Recommendations
- Based on the feature importance and model analysis, we provide the following recommendations for optimizing shrimp farming:
  - Maintain optimal water quality parameters such as temperature, dissolved oxygen, and pH.
  - Ensure regular and accurate feed management to improve growth rates.
  - Monitor and control shrimp health to reduce mortalities and enhance survival rates.
  - Utilize high-quality seed and maintain appropriate stocking densities.
