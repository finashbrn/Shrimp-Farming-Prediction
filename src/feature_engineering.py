import pandas as pd
import numpy as np

def load_and_clean_data():
    # Load the datasets
    cycles_df = pd.read_csv('data/cycles.csv')
    cycles_df = cycles_df.rename(columns={'id':'cycle_id'})
    farms_df = pd.read_csv('data/farms.csv')

    fasting_df1 = pd.read_csv('data/fasting.csv')
    fasting_df2 = pd.read_csv('data/fastings.csv')
    fasting_df = pd.concat([fasting_df1, fasting_df2], axis=0).reset_index(drop=True)
    fasting_df['cycle_id'] = fasting_df['cycle_id'].astype(np.int64).astype(str).str.strip()
    fasting_df = fasting_df.drop_duplicates().reset_index(drop=True)

    feed_tray_df = pd.read_csv('data/feed_tray.csv')
    feeds_df = pd.read_csv('data/feeds.csv')
    harvests_df = pd.read_csv('data/harvests.csv')
    measurements_df = pd.read_csv('data/measurements.csv')
    mortalities_df = pd.read_csv('data/mortalities.csv')
    ponds_df = pd.read_csv('data/ponds.csv')
    samplings_df = pd.read_csv('data/samplings.csv')

    # Handle missing values
    def handle_missing_values(df):
        for column in df.select_dtypes(include=[np.number]).columns:
            df[column].fillna(df[column].median(), inplace=True)
        for column in df.select_dtypes(include=[object]).columns:
            df[column].fillna(df[column].mode()[0], inplace=True)
        return df

    datasets = [cycles_df, farms_df, fasting_df, feed_tray_df, feeds_df, harvests_df,
                measurements_df, mortalities_df, ponds_df, samplings_df]

    cleaned_datasets = [handle_missing_values(df) for df in datasets]

    # Convert date columns to datetime
    date_columns = {
        'cycles_df': ['started_at', 'finished_at', 'created_at', 'updated_at', 'ordered_at', 'extracted_at'],
        'feed_tray_df': ['logged_at', 'feed_logged_at','created_at', 'updated_at', 'local_feed_logged_at'],
        'feeds_df': ['logged_at'],
        'harvests_df': ['harvested_at', 'created_at', 'updated_at'],
        'measurements_df': ['measured_date'],
        'mortalities_df': ['recorded_at', 'created_at', 'updated_at'],
        'ponds_df': ['created_at', 'updated_at', 'extracted_at'],
        'samplings_df': ['updated_at', 'sampled_at', 'created_at']
    }

    for df_name, cols in date_columns.items():
        for col in cols:
            eval(df_name)[col] = pd.to_datetime(eval(df_name)[col], errors='coerce')

    return cleaned_datasets

def feature_engineering(datasets):
    cycles_df, farms_df, fasting_df, feed_tray_df, feeds_df, harvests_df, \
    measurements_df, mortalities_df, ponds_df, samplings_df = datasets

    # Time features
    cycles_df['duration_days'] = (cycles_df['finished_at'] - cycles_df['started_at']).dt.days
    cycles_df['season_started'] = cycles_df['started_at'].dt.month % 12 // 3 + 1
    cycles_df['season_finished'] = cycles_df['finished_at'].dt.month % 12 // 3 + 1

    # Aggregated Feed and Fasting Metrics
    feed_agg = feeds_df.groupby('cycle_id').agg({'quantity': ['sum', 'mean', 'std']}).reset_index()
    feed_agg.columns = ['cycle_id', 'feed_sum', 'feed_mean', 'feed_std']

    fasting_agg = fasting_df.groupby('cycle_id').agg({'fasting': ['sum', 'mean']}).reset_index()
    fasting_agg.columns = ['cycle_id', 'fasting_days', 'fasting_mean']

    # Water Quality Metrics
    measurement_agg = measurements_df.groupby('cycle_id').agg({
        'morning_temperature': ['mean', 'std'],
        'evening_temperature': ['mean', 'std'],
        'morning_do': ['mean', 'std'],
        'evening_do': ['mean', 'std'],
        'morning_salinity': ['mean', 'std'],
        'evening_salinity': ['mean', 'std'],
        'morning_pH': ['mean', 'std'],
        'evening_pH': ['mean', 'std']
    }).reset_index()
    measurement_agg.columns = [
        'cycle_id', 'morning_temp_mean', 'morning_temp_std',
        'evening_temp_mean', 'evening_temp_std', 'morning_do_mean', 'morning_do_std',
        'evening_do_mean', 'evening_do_std', 'morning_salinity_mean', 'morning_salinity_std',
        'evening_salinity_mean', 'evening_salinity_std', 'morning_pH_mean', 'morning_pH_std',
        'evening_pH_mean', 'evening_pH_std'
    ]

    # Spatial Features
    farms_df['province_encoded'] = farms_df['province'].astype('category').cat.codes
    farms_df['regency_encoded'] = farms_df['regency'].astype('category').cat.codes

    ponds_df['pond_size'] = ponds_df['length'] * ponds_df['width'] * ponds_df['deep']

    # Biological Metrics
    samplings_df['prev_weight'] = samplings_df.groupby('cycle_id')['average_weight'].shift(1)
    samplings_df['prev_date'] = samplings_df.groupby('cycle_id')['sampled_at'].shift(1)
    samplings_df['days'] = (samplings_df['sampled_at'] - samplings_df['prev_date']).dt.days
    samplings_df['ADG'] = (samplings_df['average_weight'] - samplings_df['prev_weight']) / samplings_df['days']
    adg_df = samplings_df.groupby('cycle_id')['ADG'].mean().reset_index()

    # Merge all features
    features_df = cycles_df.merge(feed_agg, on='cycle_id', how='left') \
                           .merge(fasting_agg, on='cycle_id', how='left') \
                           .merge(measurement_agg, on='cycle_id', how='left') \
                           .merge(ponds_df[['id', 'pond_size','farm_id']].rename(columns={'id':'pond_id'}), on='pond_id', how='left') \
                           .merge(farms_df[['id', 'province_encoded', 'regency_encoded']].rename(columns={'id':'farm_id'}), on='farm_id', how='left') \
                           .merge(adg_df, on='cycle_id', how='left')

    # Fill any remaining missing values
    features_df = features_df.fillna(0)

    # Save the feature engineered data for further use
    features_df.to_csv('features_df.csv', index=False)
    print("Feature engineering completed and saved to features_df.csv")

if __name__ == "__main__":
    datasets = load_and_clean_data()
    feature_engineering(datasets)
