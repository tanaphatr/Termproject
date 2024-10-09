import pandas as pd

def preprocess_data(df):
    # Convert sale_date to datetime, coerce errors
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')

    # Create day_of_week feature
    df['day_of_week'] = df['sale_date'].dt.dayofweek

    # Create is_weekend feature
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Clean weather column names
    df.columns = df.columns.str.strip().str.replace('Clooudy', 'Cloudy').str.replace('Showe', 'Showers')

    # Convert event, festival, and weather to numeric codes
    df['event'], _ = pd.factorize(df['event'])
    df['festival'], _ = pd.factorize(df['festival'])
    df['weather'], _ = pd.factorize(df['weather'])

    # Fill missing values with 0 only for numeric columns
    numeric_columns = df.select_dtypes(include='number').columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    df.to_csv('cleaned_data.csv', index=False, encoding='utf-8-sig')
    return df
