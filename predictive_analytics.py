import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def log_message(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{current_time} - {message}")

def load_engagement_data(file_path):
    data = pd.read_csv(file_path)
    log_message(f"Loaded engagement data from {file_path}.")
    return data

def predict_engagement_trend(X, y, future_periods):
    model = LinearRegression()
    model.fit(X, y)
    log_message("Fitted linear regression model on engagement data.")
    
    predictions = model.predict(future_periods)
    log_message("Predicted future engagement rates.")
    return predictions

def save_predictions(predictions, output_file):
    pd.DataFrame(predictions, columns=['Predicted Engagement']).to_csv(output_file, index=False)
    log_message(f"Saved predicted engagement data to {output_file}.")

if __name__ == "__main__":
    data_file = 'engagement_data.csv'
    data = load_engagement_data(data_file)
    
    historical_data = data[['Day']].values
    engagement_rates = data['Engagement'].values
    
    future_periods = np.array([[6], [7], [8], [9], [10]])
    predictions = predict_engagement_trend(historical_data, engagement_rates, future_periods)
    
    print(f"Predicted engagement rates for future days: {predictions}")
    
    save_predictions(predictions, 'predicted_engagement.csv')
    log_message("Engagement forecasting process completed.")
