import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import joblib
from numpy import asarray
from datetime import datetime
import matplotlib.dates as mdates
# Load the data from OpenWeatherMap's history data
df = pd.read_csv('vancouver_weather.csv')

# Initialize the LabelEncoder
le_weather = LabelEncoder()
# Fit and transform the weather data
# Create a list of all possible labels
all_labels = ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist', 'Rain', 'Smoke', 'Snow', 'Thunderstorm','Dust','Sand','Ash','Squall','Tornado']
le_weather.fit(all_labels)
#print(le_weather.classes_) # print the unique labels
df['Weather'] = le_weather.transform(df['Weather'])

def train_and_evaluate(df, features, target):
    X = []
    y = []
    for i in range(24, len(df)):
        X.append(df[features][i-24:i].values.flatten())
        y.append(df[target][i])
    X = np.array(X)
    y = np.array(y)
    # Define the size of the test set
    test_size = 2400  

    X_train = X[:-test_size]
    y_train = y[:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    # Choose the model based on the target variable
    if target.lower() == 'weather':
        # Classification case
        model = RandomForestClassifier(n_estimators=100, min_samples_split=2)  
    else:
        # Regression case
        model = RandomForestRegressor(n_estimators=100, min_samples_split=2)  

    model.fit(X_train, y_train)
    
    # Visualize decision trees
    plt.figure(figsize=(20, 20))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plot_tree(model.estimators_[i], filled=True, rounded=True, max_depth=5, fontsize=8)
        plt.title(f'Tree {i+1}')
     
    plt.show()
    
    # Make predictions with the model
    y_pred = model.predict(X_test)
    print('Testing results:')
    
    if target.lower() == 'weather':
        # Classification case
        print(metrics.classification_report(y_test, y_pred, zero_division=1))
    else:
        # Regression case
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', rmse)
        
    # Predict next 24 hours weather
    if target.lower() == 'weather':
        y_pred = le_weather.inverse_transform(y_pred.astype(int))
    print(f'Predicted {target} for Next 24 hours:', y_pred)  

    # Save the model to a file
    joblib.dump(model, f'{target}.joblib')

#input the target and features you want to train
feature = ["Weather"]
target = "Weather"

# call the funtion
train_and_evaluate(df, feature, target)

#Validating with valid set
# Load the Validation datasets
new_df = pd.read_csv('validation_set.csv')
new_df['Weather'] = le_weather.transform(new_df['Weather'])

# Define your variables
one_day = 24 * 2
one_month = 24 * 31
six_months = 24 * 181
one_year = 24 * 366

# Choose the variable you want to use
prediction_length = six_months  # Change this to six_months or one_year if you want

# Define a dictionary to map the target variable to its unit
units = {'Temperature': '°C', 'Humidity': '%', 'Pressure': 'hPa'}

# Convert your date column to datetime
new_df['Date'] = pd.to_datetime(new_df['Date'])

# Load the trained model
model = joblib.load(f'{target}.joblib')

# Initialize empty lists for storing predictions and actual values
y_predicted = []
y_true = []

# Use a sliding window approach for prediction
for i in range(24, prediction_length):  # Only loop over the first prediction_length hours
    # Use the previous 24 hours of data to predict the next hour
    X_predicted = new_df[feature].values[i-24:i].flatten()
    y_true.append(new_df[target].values[i])
    y_predicted.append(model.predict([X_predicted])[0])
    X_predicted = np.append(X_predicted[1:], y_predicted[-1])

# Convert lists to numpy arrays
y_predicted = np.array(y_predicted)
y_true = np.array(y_true)

print('Validating results:')
if target.lower() == 'weather':
    # Classification case
    print(metrics.classification_report(y_true, y_predicted, zero_division=1))
else:
    # Calculate RMSE
    rmse = mean_squared_error(y_true, y_predicted, squared=False)
    print(f'RMSE for {target} predictions: {rmse:.2f}')

    # Create line plot
    plt.figure(figsize=(10, 6))
    plt.plot(new_df['Date'][24:prediction_length], y_predicted, label='Predicted', marker='o', alpha=0.5)
    plt.plot(new_df['Date'][24:prediction_length], y_true, label='Actual', marker='o', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel(f'{target} ({units[target]})')
    plt.title(f'{target} Predictions vs Actual Values for Next {(prediction_length//24)-1} Days')
    plt.legend()

    # Format x-axis to properly display dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) #adjust it to change the date display on the graph
    #plt.gca().xaxis.set_major_locator(mdates.HourLocator()) #display date label in hourly
    #plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1)) # display date label in daily
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator()) # display date label in monthly
    plt.gcf().autofmt_xdate()
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.show()
# print predicted result
if target == 'Weather':
    y_predicted = le_weather.inverse_transform(y_predicted.astype(int))
print(f'Predicted {target} for Next {(prediction_length//24)-1} Days:', y_predicted)