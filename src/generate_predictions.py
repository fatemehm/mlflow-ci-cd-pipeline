import os
import logging
from pyspark.sql import SparkSession
import mlflow.pyfunc
import pandas as pd

# Setup logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))
logger = logging.getLogger(__name__)

def load_data(spark, path):
    """Load Parquet file into Spark DataFrame."""
    try:
        return spark.read.parquet(path).select("review", "label")
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        raise

def generate_predictions(spark, input_path, output_path, model_uri="models:/SentimentModel/1"):
    """Generate predictions using MLflow PyFunc model."""
    try:
        # Load Spark DataFrame
        df_spark = load_data(spark, input_path)

        # Load MLflow PyFunc model
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        # Prepare input for the model - use 'text' column name as expected by the model
        reviews_df = pd.DataFrame({"text": [row.review for row in df_spark.collect()]})

        # Get predictions from MLflow model
        pred_df = loaded_model.predict(reviews_df)
        
        # Get original labels for comparison
        original_labels = [row.label for row in df_spark.collect()]
        original_reviews = [row.review for row in df_spark.collect()]

        # Map predictions to expected format
        predictions = []
        
        # Handle different possible output formats from the model
        if isinstance(pred_df, pd.DataFrame):
            # If pred_df is a DataFrame, iterate through rows
            for i, (review, label) in enumerate(zip(original_reviews, original_labels)):
                pred_row = pred_df.iloc[i]
                if 'label' in pred_row and 'score' in pred_row:
                    # Standard transformers output format
                    predictions.append({
                        "review": review,
                        "target": label,
                        "prediction": pred_row['label'],
                        "confidence": pred_row['score']
                    })
                else:
                    # Handle other possible formats
                    # Look for prediction columns (common names)
                    pred_value = None
                    confidence_value = None
                    
                    for col in pred_row.index:
                        if col.lower() in ['prediction', 'predicted_label', 'class']:
                            pred_value = pred_row[col]
                        elif col.lower() in ['confidence', 'probability', 'score']:
                            confidence_value = pred_row[col]
                    
                    predictions.append({
                        "review": review,
                        "target": label,
                        "prediction": pred_value if pred_value is not None else 0,
                        "confidence": confidence_value if confidence_value is not None else 0.0
                    })
        else:
            # If pred_df is a list or array
            for review, label, pred in zip(original_reviews, original_labels, pred_df):
                if isinstance(pred, dict):
                    predictions.append({
                        "review": review,
                        "target": label,
                        "prediction": pred.get("label", pred.get("prediction", 0)),
                        "confidence": pred.get("score", pred.get("confidence", 0.0))
                    })
                else:
                    # Simple prediction value
                    predictions.append({
                        "review": review,
                        "target": label,
                        "prediction": pred,
                        "confidence": 0.0  # No confidence available
                    })

        # Save predictions to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(predictions).to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")

    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        raise

def main():
    """Generate reference and current predictions."""
    spark = SparkSession.builder.appName("GeneratePredictions") \
        .config("spark.eventLog.enabled", "false").getOrCreate()
    try:
        generate_predictions(spark, "data/test.parquet", "data/reference_predictions.csv")
        generate_predictions(spark, "data/train.parquet", "data/predictions.csv")
    finally:
        spark.stop()

if __name__ == "__main__":
    main()