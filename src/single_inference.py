import re
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load the trained model and tokenizer
model_path = 'models/sentiment_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Sample reviews to test
reviews = [
    "This movie is absolutely fantastic and thrilling!",
    "The plot was boring and predictable.",
    "An amazing experience, loved every minute!"
]

# Preprocess and predict
for review in reviews:
    # Preprocess the review (mimic preprocessing.py)
    clean_review = review.lower()
    clean_review = re.sub(r'[^a-z\s]', '', clean_review)
    
    # Perform inference
    result = classifier(clean_review)
    # Debug: Print raw model output
    print(f"Raw result: {result}")
    
    # Map labels correctly
    label = 'positive' if result[0]['label'] == 'POSITIVE' else 'negative'
    score = result[0]['score']
    
    print(f"Review: {review}")
    print(f"Cleaned Review: {clean_review}")
    print(f"Predicted Sentiment: {label} (confidence: {score:.3f})\n")