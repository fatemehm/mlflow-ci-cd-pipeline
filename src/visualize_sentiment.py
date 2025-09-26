from pyspark.sql import SparkSession
import matplotlib.pyplot as plt

# Initialize Spark with event logging disabled
spark = SparkSession.builder \
    .appName('Visualize') \
    .config("spark.eventLog.enabled", "false") \
    .getOrCreate()

# Load inference results
df = spark.read.parquet('data/inference_results.parquet')
sentiment_counts = df.groupBy('predicted_sentiment').count().collect()

# Prepare data for plotting
sentiments = [row['predicted_sentiment'] for row in sentiment_counts]
counts = [row['count'] for row in sentiment_counts]

# Create bar chart
plt.figure(figsize=(8, 6))
plt.bar(sentiments, counts, color=['gray', 'green', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution in Inference Results')
plt.tight_layout()

# Save chart to file
plt.savefig('sentiment_distribution.png')
plt.close()

print("Chart saved as 'sentiment_distribution.png'")
spark.stop()