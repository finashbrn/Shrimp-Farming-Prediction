import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle

# Load the feature engineered data
features_df = pd.read_csv('features_df.csv')

# Survival Rate
sr_df = pd.read_csv('SR_datasets.csv')
sr_features_df = features_df.merge(sr_df, on='cycle_id', how='left')
sr_features_df = sr_features_df[sr_features_df['SR'].notna()]

# Average Body Weight
abw_df = pd.read_csv('ADG_datasets.csv')
abw_features_df = features_df.merge(abw_df, on='cycle_id', how='left')

# Biomass
harvests_df = pd.read_csv('harvests.csv')
harvests_df['biomass'] = harvests_df['weight']
biomass_df = harvests_df.groupby('cycle_id')['biomass'].sum().reset_index()
biomass_features_df = features_df.merge(biomass_df, on='cycle_id', how='left')

# Revenue
harvests_df['revenue'] = harvests_df['weight'] * harvests_df['selling_price']
revenue_df = harvests_df.groupby('cycle_id')['revenue'].sum().reset_index()
revenue_features_df = features_df.merge(revenue_df, on='cycle_id', how='left')

def train_model(features_df, target_column, model_name):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_col = features_df.select_dtypes(include=numerics).columns
    num_col = num_col.to_list()
    num_col.append('cycle_id')

    features_df = features_df[num_col]
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.dropna(inplace=True)

    X = features_df.drop(columns=['cycle_id', target_column])
    y = features_df[target_column]

    X = X.apply(lambda x: np.where(np.abs(x) > np.finfo(np.float32).max, np.nan, x))
    X.dropna(inplace=True)
    y = y[X.index]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    # Hyperparameter tuning for RandomForest
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    grid_search_rf = GridSearchCV(estimator=models['Random Forest'], param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
    grid_search_rf.fit(X_train, y_train)
    best_rf = grid_search_rf.best_estimator_

    # Hyperparameter tuning for XGBoost
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    grid_search_xgb = GridSearchCV(estimator=models['XGBoost'], param_grid=param_grid_xgb, cv=3, n_jobs=-1, verbose=2)
    grid_search_xgb.fit(X_train, y_train)
    best_xgb = grid_search_xgb.best_estimator_

    # Fit and evaluate models
    model_performance = {}
    for name, model in zip(['Linear Regression', 'Random Forest', 'XGBoost'],
                           [models['Linear Regression'], best_rf, best_xgb]):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        model_performance[name] = mse
        print(f"{model_name} Model Performance ({name} - MSE): {mse}")

    # Save the best model
    if model_name == 'Survival Rate':
        best_model = best_rf
        with open('models/rf_sr_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
    elif model_name == 'Average Body Weight':
        best_model = best_rf
        with open('models/rf_abw_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
    elif model_name == 'Biomass':
        best_model = best_xgb
        with open('models/xgb_biomass_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
    elif model_name == 'Revenue':
        best_model = best_xgb
        with open('models/xgb_revenue_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)

if __name__ == "__main__":
    train_model(sr_features_df, 'SR', 'Survival Rate')
    train_model(abw_features_df, 'ABW', 'Average Body Weight')
    train_model(biomass_features_df, 'biomass', 'Biomass')
    train_model(revenue_features_df, 'revenue', 'Revenue')
