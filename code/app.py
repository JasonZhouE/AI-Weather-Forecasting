# Import the necessary modules
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import pytz
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

# Create a TimezoneFinder object
tf = TimezoneFinder()
geolocator = Nominatim(user_agent="city_timezone")

def get_geocode(city_name):
    location = geolocator.geocode(city_name)
    if location is not None:
        return (location.latitude, location.longitude)
    else:
        return None
    
# Define a function to get city name from user and its history weather data from OpenWeatherMap API
def get_weather_data():
    while True:
        city = input("Enter a city name,if there is other city has the same name, plase add the contry code \n after a comma,for example {vancouver,ca} and {vancouver,us}：")
        # Enter your API key, copied from the OpenWeatherMap dashboard
        api_key = "4147f3c8808212541e9f46913b6eed95"
        # Get the history weather data for the last 7 days in JSON format
        url = f"http://history.openweathermap.org/data/2.5/history/city?q={city}&type=hour&units=metric&appid={api_key}"
        data = requests.get(url).json()

        # Check if the 'cod' field is present in the returned JSON object
        if 'cod' not in data:
            print(f"Error: Could not find weather data for {city}")
            continue

        return city,data


# Define a function to make weather predictions using the models
def make_predictions(data):
    # Load the models from the files
    weather_model = joblib.load('Weather.joblib')
    temp_model = joblib.load('Temperature.joblib')
    humid_model = joblib.load('Humidity.joblib')
    pressure_model = joblib.load('Pressure.joblib')
    # Convert the JSON data into a pandas dataframe
    df = pd.DataFrame(data['list'])
   # Create an instance of the LabelEncoder class
    le_weather = LabelEncoder()
    all_labels = ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Thunderstorm','Dust','Sand','Ash','Squall','Tornado']
    le_weather.fit(all_labels)
    # Encode the values in the 'main' column of the 'weather' column
    weather_values_encoded = le_weather.transform([hour[0]['main'] for hour in df['weather']])

    # Append the encoded values to a new column in the df
    df['weather_encoded'] = weather_values_encoded

    # Extract only the temperature and humidity columns and convert them to numpy arrays
    weather_values = []
    temp_values = []
    humid_values = []
    pressure_values = []
    for hour in df['weather_encoded']:
        weather_values.append(hour)
    for hour in df['main']:
        temp_values.append(hour['temp'])
        humid_values.append(hour['humidity'])
        pressure_values.append(hour['pressure'])
    weather_values = np.array(weather_values)
    temp_values = np.array(temp_values)
    humid_values = np.array(humid_values)
    pressure_values = np.array(pressure_values)
    # Get the last 24 hours weather values as input for the models
    weather_row = weather_values[-24:].flatten()
    temp_row = temp_values[-24:].flatten()
    humid_row = humid_values[-24:].flatten()
    pressure_row = pressure_values[-24:].flatten()
    # Make predictions for the next 12 hours using the models
    weather_future = weather_row
    temp_future = temp_row
    humid_future = humid_row
    pressure_future = pressure_row
    weather_prediction = []
    temp_prediction = []
    humid_prediction = []
    pressure_prediction = []
    for i in range(12):
        weather_prediction.append(weather_model.predict([weather_future])[0])
        temp_prediction.append(temp_model.predict([temp_future])[0])
        humid_prediction.append(humid_model.predict([humid_future])[0])
        pressure_prediction.append(pressure_model.predict([pressure_future])[0])
        weather_future = np.append(weather_future[1:], weather_prediction[-1])
        temp_future = np.append(temp_future[1:], temp_prediction[-1])
        humid_future = np.append(humid_future[1:], humid_prediction[-1])
        pressure_future = np.append(pressure_future[1:], pressure_prediction[-1])
    weather_prediction = np.array(weather_prediction)
    weather_prediction = le_weather.inverse_transform(weather_prediction.astype(int))
    temp_prediction = np.array(temp_prediction)
    humid_prediction = np.array(humid_prediction)
    pressure_prediction = np.array(pressure_prediction)
    return weather_prediction, temp_prediction, humid_prediction, pressure_prediction

# Define a function to display the prediction result
def display_result(city, weather_prediction, temp_prediction, humid_prediction, pressure_prediction):
    # Get the time zone for the city
    geocode = get_geocode(city)
    timezone_str = tf.timezone_at(lng=geocode[1], lat=geocode[0])
    current_tz = pytz.timezone(timezone_str)
    current_time = datetime.now(current_tz)
    current_time = current_time.strftime("%Y-%m-%d %H:%M %A")
    print(f"{current_time} {city}")
    print(f"The weather prediction for {city} for the next 12 hours is:")
    print(f"{'Datetime':<12}{'Weather':<12}{'Temperature':<12}{'Humidity':<12}{'Pressure':<12}")
    # Get the last timestamp from the JSON data and convert it to a datetime object
    last_timestamp = data['list'][-1]['dt']
    
    # Add one hour for each prediction and format it as YYYY-MM-DD HH:MM:SS
    for i in range(12):
        dt_object = datetime.fromtimestamp(last_timestamp) + timedelta(hours=i+1)
        # convert the datetime object to the America/Phoenix timezone
        dt_object_tz = dt_object.astimezone(current_tz)
        dt_string = dt_object_tz.strftime("%H:%M:%S")
        # Format the prediction with one decimal place and print it with the datetime string
        print(f"{dt_string:<12}{weather_prediction[i]:<12}{format(temp_prediction[i], '.1f') + ' °C':<12}{format(humid_prediction[i], '.1f') + ' %':<12}{format(pressure_prediction[i], '.1f') + ' hPa':<12}")


# Main program
if __name__ == "__main__":
    # Get the weather data from OpenWeatherMap API
    city, data = get_weather_data()
    # Make weather predictions using the models
    weather_prediction, temp_prediction, humid_prediction, pressure_prediction = make_predictions(data)
    # Display the prediction result
    display_result(city, weather_prediction, temp_prediction, humid_prediction, pressure_prediction)