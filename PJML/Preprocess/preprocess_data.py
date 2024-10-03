# Preprocess/preprocess_data.py
import pandas as pd

def preprocess_data(df):
    # Convert sale_date to datetime, coerce errors
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')

    # Create day_of_week feature
    df['day_of_week'] = df['sale_date'].dt.dayofweek

    # Create is_weekend feature
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Create dummy variables for event, festival, and weather
    df = pd.get_dummies(df, columns=['event', 'festival', 'weather'], drop_first=True)

    # Fill missing values with 0 only for numeric columns
    numeric_columns = df.select_dtypes(include='number').columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    return df
