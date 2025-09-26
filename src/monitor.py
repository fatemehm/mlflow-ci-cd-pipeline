import pandas as pd
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
import logging
import os

# Setup logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"))
logger = logging.getLogger(__name__)

def load_data():
    """Load reference and current prediction data."""
    try:
        reference_data = pd.read_csv("data/reference_predictions.csv")
        current_data = pd.read_csv("data/predictions.csv")
        return reference_data, current_data
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def preprocess_data(df):
    """Preprocess data to ensure consistent data types."""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert prediction and target columns to consistent types
    # First, let's handle string labels that might come from transformers models
    if df['prediction'].dtype == 'object':
        # Map common string labels to integers
        label_mapping = {
            'POSITIVE': 1, 'positive': 1, 'pos': 1, 'POS': 1,
            'NEGATIVE': 0, 'negative': 0, 'neg': 0, 'NEG': 0,
            'LABEL_1': 1, 'LABEL_0': 0,
            '1': 1, '0': 0
        }
        df['prediction'] = df['prediction'].map(label_mapping).fillna(df['prediction'])
    
    # Convert to numeric, coercing any remaining strings to NaN, then fill with 0
    df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce').fillna(0).astype(int)
    df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int)
    
    # Ensure confidence is numeric
    if 'confidence' in df.columns:
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.0)
    
    return df

def generate_report(reference_data, current_data):
    """Generate data drift and classification quality report."""
    try:
        # Preprocess data to ensure consistent types
        reference_data = preprocess_data(reference_data)
        current_data = preprocess_data(current_data)
        
        # Print data info for debugging
        print("Reference data dtypes:")
        print(reference_data.dtypes)
        print("\nReference prediction unique values:", reference_data['prediction'].unique())
        print("Reference target unique values:", reference_data['target'].unique())
        
        print("\nCurrent data dtypes:")
        print(current_data.dtypes)
        print("\nCurrent prediction unique values:", current_data['prediction'].unique())
        print("Current target unique values:", current_data['target'].unique())
        
        # Define column mapping for Evidently
        column_mapping = ColumnMapping(
            target='target',
            prediction='prediction',
            text_features=['review'] if 'review' in reference_data.columns else None
        )
        
        # Create report with proper column mapping
        report = Report(metrics=[
            DataDriftPreset(),
            ClassificationPreset()
        ])
        
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        os.makedirs("reports", exist_ok=True)
        report.save_html("reports/sentiment_drift_report.html")
        logger.info("Report saved to reports/sentiment_drift_report.html")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()
        raise

def main():
    """Main function to run monitoring."""
    try:
        reference_data, current_data = load_data()
        generate_report(reference_data, current_data)
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise

if __name__ == "__main__":
    main()