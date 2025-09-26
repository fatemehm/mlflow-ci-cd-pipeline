from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, FloatType
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
#from langdetect import detect
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_text(text):
    try:
        if text is None:
            return ""
        #if detect(text) != 'en':
            #return ""  # Skip non-English
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return ""

def predict_sentiment(text):
    try:
        if not text:
            return "unknown", 0.0
        result = classifier(text)
        label = 'positive' if result[0]['label'] == 'POSITIVE' else 'negative'
        score = result[0]['score']
        # Flag low-confidence predictions
        if score < 0.6:
            label = 'uncertain'
        return label, score
    except Exception as e:
        logger.error(f"Prediction error for text '{text[:50]}...': {str(e)}")
        return "error", 0.0

# Initialize Spark
spark = SparkSession.builder \
    .appName("BatchInference") \
    .config("spark.eventLog.enabled", "false") \
    .getOrCreate()

# Load the trained model and tokenizer
try:
    model_path = 'models/sentiment_model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    spark.stop()
    raise

# Register UDFs
preprocess_udf = udf(preprocess_text, StringType())
predict_udf = udf(lambda text: predict_sentiment(text)[0], StringType())
score_udf = udf(lambda text: predict_sentiment(text)[1], FloatType())

# Load input data
input_path = 'data/test.parquet'  # Replace with new data if available
try:
    df = spark.read.parquet(input_path)
    logger.info(f"Loaded {df.count()} reviews from {input_path}")
except Exception as e:
    logger.error(f"Failed to load data: {str(e)}")
    spark.stop()
    raise

# Preprocess and predict
df = df.withColumn('clean_review', preprocess_udf(col('review')))
df = df.withColumn('predicted_sentiment', predict_udf(col('clean_review')))
df = df.withColumn('confidence_score', score_udf(col('clean_review')))

# Save results
output_path = 'data/inference_results.parquet'
try:
    df.select('review', 'clean_review', 'predicted_sentiment', 'confidence_score').write.mode('overwrite').parquet(output_path)
    logger.info(f"Results saved to {output_path}")
except Exception as e:
    logger.error(f"Failed to save results: {str(e)}")
    spark.stop()
    raise

# Show sample results
df.select('review', 'predicted_sentiment', 'confidence_score').show(5, truncate=False)

# Count predictions
print(f"Processed {df.count()} reviews.")
spark.stop()