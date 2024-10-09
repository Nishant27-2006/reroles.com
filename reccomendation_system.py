import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_user_profile(file_path):
    user_profile = pd.read_csv(file_path)
    logging.info(f"Loaded user profile data from {file_path}.")
    return user_profile

def load_content_library(file_path):
    content_library = pd.read_csv(file_path)
    logging.info(f"Loaded content library data from {file_path}.")
    return content_library

def recommend_content(user_vector, content_vectors):
    similarities = cosine_similarity(user_vector, content_vectors)
    recommended_index = np.argmax(similarities)
    logging.info(f"Content with index {recommended_index} has been recommended.")
    return recommended_index

def save_recommendation(content_index, output_file):
    with open(output_file, 'w') as f:
        f.write(f"Recommended Content Index: {content_index}\n")
    logging.info(f"Saved recommended content index to {output_file}.")

if __name__ == "__main__":
    user_profile_file = 'user_profile.csv'
    content_library_file = 'content_library.csv'
    
    user_profile = load_user_profile(user_profile_file).values
    content_library = load_content_library(content_library_file).values
    
    recommended_content_index = recommend_content(user_profile, content_library)
    print(f"Recommended content index: {recommended_content_index}")
    
    save_recommendation(recommended_content_index, 'recommended_content.txt')
