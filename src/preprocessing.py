import yaml
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover

def preprocess_data(config):
    try:
        spark = SparkSession.builder.appName("SentimentPreprocessing") \
                            .config("spark.eventLog.enabled", "false").getOrCreate()

        input_path = config['dataset']['output_path'].replace('.csv', '.parquet')
        df = spark.read.parquet(input_path)

        # Clean text
        df = df.withColumn('clean_review', regexp_replace(lower(col('review')), r'[^a-z\s]', ''))

        # Tokenize and remove stopwords
        df = StopWordsRemover(inputCol='tokens', outputCol='filtered_tokens') \
             .transform(Tokenizer(inputCol='clean_review', outputCol='tokens').transform(df))

        # Split and save
        train, test = df.randomSplit([0.8, 0.2], seed=42)
        train.write.mode('overwrite').parquet('data/train.parquet')
        test.write.mode('overwrite').parquet('data/test.parquet')

        print(f"Preprocessed {train.count()} training and {test.count()} test samples.")
        spark.stop()

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

if __name__ == "__main__":
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    preprocess_data(config)
