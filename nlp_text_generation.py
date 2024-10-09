from transformers import pipeline
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def log_message(message):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{current_time} - {message}")

def generate_social_media_caption(prompt, max_len=50, num_options=3):
    text_generator = pipeline('text-generation', model='gpt2')
    log_message("Initialized GPT-2 model for caption generation.")
    
    generated_captions = text_generator(prompt, max_length=max_len, num_return_sequences=num_options)
    log_message("Generated captions based on input prompt.")
    
    captions = [caption['generated_text'] for caption in generated_captions]
    return captions

def save_captions(captions, output_file):
    with open(output_file, 'w') as f:
        for idx, caption in enumerate(captions):
            f.write(f"Option {idx + 1}: {caption}\n")
    log_message(f"Saved generated captions to {output_file}.")

if __name__ == "__main__":
    prompt = "Create a captivating and trendy caption for a luxury fashion brand:"
    captions = generate_social_media_caption(prompt)
    
    log_message("Starting the caption generation workflow.")
    for i, caption in enumerate(captions, start=1):
        print(f"Caption Option {i}: {caption}")
    
    save_captions(captions, 'generated_captions.txt')
    log_message("Caption generation process completed.")
