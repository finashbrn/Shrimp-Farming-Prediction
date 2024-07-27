import pandas as pd
import numpy as np
import pickle

def load_models():
    with open('models/rf_sr_model.pkl', 'rb') as f:
        rf_sr_model = pickle.load(f)
    with open('models/rf_abw_model.pkl', 'rb') as f:
        rf_abw_model = pickle.load(f)
    with open('models/xgb_biomass_model.pkl', 'rb') as f:
        xgb_biomass_model = pickle.load(f)
    with open('models/xgb_revenue_model.pkl', 'rb') as f:
        xgb_revenue_model = pickle.load(f)
    return rf_sr_model, rf_abw_model, xgb_biomass_model, xgb_revenue_model

def make_predictions(features_df, rf_sr_model, rf_abw_model, xgb_biomass_model, xgb_revenue_model):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_col = features_df.select_dtypes(include=numerics).columns
    num_col = num_col.to_list()
    num_col.append('cycle_id')
    features_df = features_df[num_col]
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.dropna(inplace=True)

    # Survival Rate Predictions
    X_sr = features_df.drop(columns=['cycle_id'])
    features_df['SR_Predicted'] = rf_sr_model.predict(X_sr)

    # Average Body Weight Predictions
    X_abw = features_df.drop(columns=['cycle_id'])
    features_df['ABW_Predicted'] = rf_abw_model.predict(X_abw)

    # Biomass Predictions
    X_biomass = features_df.drop(columns=['cycle_id'])
    features_df['Biomass_Predicted'] = xgb_biomass_model.predict(X_biomass)

    # Revenue Predictions
    X_revenue = features_df.drop(columns=['cycle_id'])
    features_df['Revenue_Predicted'] = xgb_revenue_model.predict(X_revenue)

    return features_df

if __name__ == "__main__":
    features_df = pd.read_csv('features_df.csv')
    rf_sr_model, rf_abw_model, xgb_biomass_model, xgb_revenue_model = load_models()
    predictions_df = make_predictions(features_df, rf_sr_model, rf_abw_model, xgb_biomass_model, xgb_revenue_model)

    predictions_df.to_csv('Predictions/predictions.csv', index=False)
    print("Predictions saved to Predictions/predictions.csv")
