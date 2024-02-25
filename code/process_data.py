import requests
import csv
import os
import datetime

# Define the API endpoint
api_url = "http://history.openweathermap.org/data/2.5/history/city?q=seattle,us&type=hour&units=metric&appid=4147f3c8808212541e9f46913b6eed95"

# Specify the initial start timestamp
start_timestamp = 1701482400  # Your provided timestamp 1667286000

# Define the number of hours in each segment (e.g., 7 days)
hours_per_segment = 7 * 24  # 7 days * 24 hours

# Specify the total number of segments (e.g., for 9 months)
total_segments = 52  # 52 * 7days = 364 days

# Specify the CSV file name
csv_file_name = 'validation_set.csv'

# Define the CSV header
csv_header = ['Date', 'Temperature', 'Pressure', 'Humidity', 'Wind Speed', 'Wind Degree', 'Clouds', 'Weather', 'Weather Description', 'Weather Icon', 'Rain', 'Snow']

# Initialize a list to store the weather data
weather_data_list = []

# Loop through the segments and fetch data in each segment
for segment in range(total_segments):
    # Calculate the end timestamp for the current segment
    end_timestamp = start_timestamp + (hours_per_segment * 3600)

    # Send a GET request to the API with the current timestamp range
    response = requests.get(f"{api_url}&start={start_timestamp}&end={end_timestamp}")

    if response.status_code == 200:
        data = response.json()
        weather_data_list.extend(data['list'])
    else:
        print("Error fetching data")

    # Update the start timestamp for the next segment
    start_timestamp = end_timestamp + 3600  # Add one hour for the next segment

# Open the CSV file in append mode, but if it doesn't exist, open it in write mode initially to write the header
mode = 'a' if os.path.exists(csv_file_name) else 'w'
with open(csv_file_name, mode, newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_header)

    # If the file is newly created, write the header
    if mode == 'w':
        writer.writeheader()

    # Write the data
    for item in weather_data_list:
        weather = item['weather'][0]
        rain = 0
        snow = 0

        # Check if 'rain' and 'snow' keys exist in the 'item' dictionary
        if 'rain' in item:
            rain = item['rain'].get('1h', 0)  # If 'rain' exists, use the '1h' key, otherwise, set it to 0
        if 'snow' in item:
            snow = item['snow'].get('1h', 0)  # If 'snow' exists, use the '1h' key, otherwise, set it to 0

        writer.writerow({
            'Date': item['dt'],
            'Temperature': item['main']['temp'],
            'Pressure': item['main']['pressure'],
            'Humidity': item['main']['humidity'],
            'Wind Speed': item['wind']['speed'],
            'Wind Degree': item['wind']['deg'],  # Add the 'Wind Degree' field
            'Clouds': item['clouds']['all'],
            'Weather': weather['main'],
            'Weather Description': weather['description'],
            'Weather Icon': weather['icon'],  # Add the 'Weather Icon' field
            'Rain': rain,
            'Snow': snow
        })

print("Data has been fetched and appended to the CSV file.")