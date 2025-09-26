import re
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load the trained model and tokenizer
model_path = 'models/sentiment_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Bench review
review = "The bench is very nice. I had a few pieces that looked like they were painted over scratches and some of the metal were scratched. I was unsure of some of the screws as you can not tighten it too much or is strips the plastic. wasnt sure on the indents on the metal bars which way they were suppose to go on. the bench broke after sitting on it three times. The glider bar broke. the weld is not very good or sturdy."

# Preprocess the review
clean_review = review.lower()
clean_review = re.sub(r'[^a-z\s]', '', clean_review)

# Perform inference
result = classifier(clean_review)
print(f"Raw result: {result}")
label = 'positive' if result[0]['label'] == 'POSITIVE' else 'negative'
score = result[0]['score']

print(f"Review: {review}")
print(f"Cleaned Review: {clean_review}")
print(f"Predicted Sentiment: {label} (confidence: {score:.3f})")