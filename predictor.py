# predictor.py

import pandas as pd
import joblib
import requests
import holidays
from datetime import datetime

# --- Configuration Constants ---
# These are loaded once when the module is imported, making it efficient.
RESTAURANT_LOCATION = "Tamil Nadu"
HOLIDAY_COUNTRY = 'IN'
HOLIDAY_PROVINCE = 'TN'
MODEL_PATH = 'xgboost_model.joblib'
COLUMNS_PATH = 'model_columns.joblib'
DATA_PATH = 'full_details_for_app.csv'
API_KEY_PATH = 'api_key.txt'

# --- Load Artifacts on Startup ---
try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    df_base = pd.read_csv(DATA_PATH)
    holiday_calendar = holidays.CountryHoliday(HOLIDAY_COUNTRY, prov=None, state=HOLIDAY_PROVINCE)
    with open(API_KEY_PATH, 'r') as f:
        api_key = f.read().strip()
except FileNotFoundError as e:
    raise RuntimeError(f"Could not initialize predictor: {e}. Make sure all required files are present.")

# --- Helper Function for Weather ---
def get_hourly_weather_forecast(location, target_date, target_hour):
    """Fetches weather forecast for a specific date and hour from WeatherAPI."""
    base_url = "http://api.weatherapi.com/v1/forecast.json"
    days = (target_date - datetime.today().date()).days + 1

    if not (0 < days <= 14):
        return {"error": "Date must be within the next 14 days."}

    params = {'key': api_key, 'q': location, 'days': days}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        hourly_data = response.json()['forecast']['forecastday'][-1]['hour']
        return hourly_data[target_hour]
    except Exception as e:
        return {"error": f"API or data parsing error: {e}"}

# --- Main Prediction Function ---
def generate_predictions(target_date_str: str, target_hour: int, is_special_event: bool):
    """
    Generates order predictions for all menu items for a given time and conditions.
    """
    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    except ValueError:
        return {"error": "Invalid date format. Please use YYYY-MM-DD."}

    # 1. Fetch weather data
    weather_data = get_hourly_weather_forecast(RESTAURANT_LOCATION, target_date, target_hour)
    if "error" in weather_data:
        return weather_data # Propagate the error message

    # 2. Prepare features for the model
    unique_items = df_base[['food_item_name', 'food_item_category']].drop_duplicates()
    day_of_week = target_date.strftime('%A')
    day_type = 'Weekend' if day_of_week in ['Saturday', 'Sunday'] else 'Weekday'
    is_holiday = target_date in holiday_calendar

    # 3. Build the scenario DataFrame
    scenarios = []
    for _, row in unique_items.iterrows():
        base_scenario = {
            'hour': target_hour,
            'food_item_name': row['food_item_name'],
            'food_item_category': row['food_item_category'],
            'day_of_the_week': day_of_week,
            'day_type': day_type,
            'temperature_c': weather_data['temp_c'],
            'wind_kph': weather_data['wind_kph'],
            'precipitation_mm': weather_data['precip_mm'],
            'cloud': weather_data['cloud'],
            'humidity': weather_data['humidity'],
            'pressure_mb': weather_data['pressure_mb'],
            'is_holiday': is_holiday,
            'is_special_event': is_special_event,
        }
        scenarios.append({**base_scenario, 'order_type': 'Dine In'})
        scenarios.append({**base_scenario, 'order_type': 'Take Away'})

    future_df = pd.DataFrame(scenarios)

    # 4. Preprocess, predict, and format
    future_encoded = pd.get_dummies(future_df, columns=['food_item_name', 'food_item_category', 'day_of_the_week', 'day_type', 'order_type'])
    future_aligned = future_encoded.reindex(columns=model_columns, fill_value=0)
    
    predictions_proba = model.predict_proba(future_aligned)[:, 1]
    future_df['probability'] = predictions_proba

    # 5. Aggregate and format results
    # Overall
    overall = future_df.groupby('food_item_name')['probability'].mean().reset_index()
    overall = overall.sort_values('probability', ascending=False)
    overall['probability'] = overall['probability'].map('{:.2%}'.format)

    # Detailed
    detailed = future_df.pivot(index='food_item_name', columns='order_type', values='probability').reset_index().fillna(0)
    detailed['Dine In'] = detailed['Dine In'].map('{:.2%}'.format)
    detailed['Take Away'] = detailed['Take Away'].map('{:.2%}'.format)

    # 6. Structure the final output
    output = {
        "prediction_metadata": {
            "target_date": target_date_str,
            "target_hour": target_hour,
            "is_special_event": is_special_event,
            "is_public_holiday": is_holiday,
            "day_of_week": day_of_week
        },
        "weather_conditions": {
            "temperature_c": weather_data['temp_c'],
            "precipitation_mm": weather_data['precip_mm'],
            "wind_kph": weather_data['wind_kph'],
            "cloud_percent": weather_data['cloud']
        },
        "overall_prediction": overall.to_dict(orient='records'),
        "detailed_prediction": detailed.to_dict(orient='records')
    }
    
    return output
