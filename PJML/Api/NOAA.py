import requests
from datetime import datetime

# Function to fetch historical weather data from NOAA CDO API
def fetch_weather_data(api_key, start_date, end_date, location_id):
    url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    params = {
        "datasetid": "GHCND",  # Global Historical Climatology Network Daily dataset
        "startdate": start_date,
        "enddate": end_date,
        "locationid": location_id,
        "limit": 1,  # Increase limit to fetch more results
        "units": "metric"  # Optional: request data in metric units
    }

    headers = {
        "Token": api_key
    }

    print("Fetching weather data...")  # Show that the fetching is in progress
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        print("Weather data fetched successfully.")  # Confirm that data was fetched
        return data
    else:
        print(f"Error: {response.status_code}")
        return None

def filter_weather_data(weather_data):
    filtered_data = []
    for item in weather_data.get('results', []):
        # Check if the data has the desired elements
        if item.get('datatype') in ['PRCP', 'SKY', 'TMAX']:  # Precipitation, Sky Cover, Max Temperature
            filtered_data.append(item)
    return filtered_data

# Usage example
api_key = "RrDROzJymwgaEjRWgbVWgmZIZXJYZygr"  # Replace with your actual API key
start_date = "2022-11-05"  # Start date in YYYY-MM-DD format
end_date = datetime.now().strftime("%Y-%m-%d")  # Current date in YYYY-MM-DD format
location_id = "FIPS:TH"  # Location ID (replace with desired location)

weather_data = fetch_weather_data(api_key, start_date, end_date, location_id)

if weather_data:
    filtered_weather_data = filter_weather_data(weather_data)
    for record in filtered_weather_data:
        print(record)  # Print filtered weather data
