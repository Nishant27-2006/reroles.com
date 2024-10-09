from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_feedback(file_path):
    feedback_data = pd.read_csv(file_path)
    logging.info(f"Loaded feedback data from {file_path}.")
    return feedback_data

def analyze_sentiment(feedback):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = feedback['Feedback'].apply(lambda x: analyzer.polarity_scores(x))
    feedback['Sentiment'] = sentiments
    logging.info("Completed sentiment analysis on feedback.")
    return feedback

def save_sentiment_results(feedback, output_file):
    feedback.to_csv(output_file, index=False)
    logging.info(f"Saved sentiment analysis results to {output_file}.")

if __name__ == "__main__":
    feedback_file = 'customer_feedback.csv'
    feedback_data = load_feedback(feedback_file)
    
    sentiment_results = analyze_sentiment(feedback_data)
    print(sentiment_results[['Feedback', 'Sentiment']])
    
    save_sentiment_results(sentiment_results, 'sentiment_analysis_results.csv')
