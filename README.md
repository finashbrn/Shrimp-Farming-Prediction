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

```
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
 
## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/finashbrn/Shrimp-Farming-Prediction.git
   cd Shrimp-Farming-Prediction
   ```
2. **Create and Activate a Virtual Environment:**
  ```bash
   python -m venv env
   env\Scripts\activate
```
3. **Create and Activate a Virtual Environment:**
 ```bash
  pip install -r requirements.txt
```
## Running the Code
 **Feature Engineering:**
```bash
   python src/feature_engineering.py
   ```
 **Model Training:**
```bash
   python src/train_model.py
   ```
 **Making Predictions:**
```bash
  python src/predict.py
   ```

## API Setup and Usage

### Prerequisites

- Python 3.6 or higher
- Flask
- pandas
- numpy
- scikit-learn
- xgboost

### Setup

1. Clone the repository and navigate to the project directory:

    ```bash
    git clone https://github.com/finashbrn/Shrimp-Farming-Prediction.git
    cd Shrimp-Farming-Prediction
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv env
    env\Scripts\activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Navigate to the `api` directory and run the Flask application:

    ```bash
    cd api
    flask run
    ```

### API Endpoints

- **Predict Survival Rate**: `POST /predict/sr`
- **Predict Average Body Weight**: `POST /predict/abw`
- **Predict Biomass**: `POST /predict/biomass`
- **Predict Revenue**: `POST /predict/revenue`

### Example Requests

#### Predict Survival Rate

```bash
curl -X POST http://127.0.0.1:5000/predict/sr -H "Content-Type: application/json" -d '{
    "duration_days": 100,
    "season_started": 2,
    "season_finished": 3,
    "morning_temp_mean": 26.5,
    "morning_temp_std": 2.1,
    "evening_temp_mean": 27.3,
    "evening_temp_std": 1.8,
    "morning_do_mean": 5.6,
    "morning_do_std": 0.7,
    "evening_do_mean": 5.8,
    "evening_do_std": 0.6,
    "morning_salinity_mean": 30.1,
    "morning_salinity_std": 1.0,
    "evening_salinity_mean": 30.4,
    "evening_salinity_std": 0.9,
    "morning_pH_mean": 7.8,
    "morning_pH_std": 0.2,
    "evening_pH_mean": 7.7,
    "evening_pH_std": 0.3,
    "feed_sum": 1200.0,
    "feed_mean": 30.0,
    "feed_std": 5.0,
    "fasting_days": 5,
    "fasting_mean": 0.1,
    "pond_size": 1500,
    "province_encoded": 3,
    "regency_encoded": 10,
    "ADG": 0.3
}'
#### Predict Average Body Weight
curl -X POST http://127.0.0.1:5000/predict/abw -H "Content-Type: application/json" -d '{
    "duration_days": 100,
    "season_started": 2,
    "season_finished": 3,
    "morning_temp_mean": 26.5,
    "morning_temp_std": 2.1,
    "evening_temp_mean": 27.3,
    "evening_temp_std": 1.8,
    "morning_do_mean": 5.6,
    "morning_do_std": 0.7,
    "evening_do_mean": 5.8,
    "evening_do_std": 0.6,
    "morning_salinity_mean": 30.1,
    "morning_salinity_std": 1.0,
    "evening_salinity_mean": 30.4,
    "evening_salinity_std": 0.9,
    "morning_pH_mean": 7.8,
    "morning_pH_std": 0.2,
    "evening_pH_mean": 7.7,
    "evening_pH_std": 0.3,
    "feed_sum": 1200.0,
    "feed_mean": 30.0,
    "feed_std": 5.0,
    "fasting_days": 5,
    "fasting_mean": 0.1,
    "pond_size": 1500,
    "province_encoded": 3,
    "regency_encoded": 10,
    "ADG": 0.3
}'

#### Predict Biomass
curl -X POST http://127.0.0.1:5000/predict/biomass -H "Content-Type: application/json" -d '{
    "duration_days": 100,
    "season_started": 2,
    "season_finished": 3,
    "morning_temp_mean": 26.5,
    "morning_temp_std": 2.1,
    "evening_temp_mean": 27.3,
    "evening_temp_std": 1.8,
    "morning_do_mean": 5.6,
    "morning_do_std": 0.7,
    "evening_do_mean": 5.8,
    "evening_do_std": 0.6,
    "morning_salinity_mean": 30.1,
    "morning_salinity_std": 1.0,
    "evening_salinity_mean": 30.4,
    "evening_salinity_std": 0.9,
    "morning_pH_mean": 7.8,
    "morning_pH_std": 0.2,
    "evening_pH_mean": 7.7,
    "evening_pH_std": 0.3,
    "feed_sum": 1200.0,
    "feed_mean": 30.0,
    "feed_std": 5.0,
    "fasting_days": 5,
    "fasting_mean": 0.1,
    "pond_size": 1500,
    "province_encoded": 3,
    "regency_encoded": 10,
    "ADG": 0.3
}'

#### Predict Revenue
curl -X POST http://127.0.0.1:5000/predict/revenue -H "Content-Type: application/json" -d '{
    "duration_days": 100,
    "season_started": 2,
    "season_finished": 3,
    "morning_temp_mean": 26.5,
    "morning_temp_std": 2.1,
    "evening_temp_mean": 27.3,
    "evening_temp_std": 1.8,
    "morning_do_mean": 5.6,
    "morning_do_std": 0.7,
    "evening_do_mean": 5.8,
    "evening_do_std": 0.6,
    "morning_salinity_mean": 30.1,
    "morning_salinity_std": 1.0,
    "evening_salinity_mean": 30.4,
    "evening_salinity_std": 0.9,
    "morning_pH_mean": 7.8,
    "morning_pH_std": 0.2,
    "pH_std": 0.2,
    "evening_pH_mean": 7.7,
    "evening_pH_std": 0.3,
    "feed_sum": 1200.0,
    "feed_mean": 30.0,
    "feed_std": 5.0,
    "fasting_days": 5,
    "fasting_mean": 0.1,
    "pond_size": 1500,
    "province_encoded": 3,
    "regency_encoded": 10,
    "ADG": 0.3
}'

