import os
import shutil
import glob
import yaml
import pandas as pd
from pyspark.sql import SparkSession
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.transformers

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {'accuracy': accuracy_score(labels, preds), 'f1': f1_score(labels, preds, average='weighted')}

def train_model(config):
    try:
        # Spark setup
        spark = SparkSession.builder.appName("SentimentTraining") \
                            .config("spark.eventLog.enabled", "false").getOrCreate()

        # Load parquet
        train_df, test_df = [spark.read.parquet(f"data/{x}.parquet") for x in ["train", "test"]]

        # Helper to write single CSV file from Spark DataFrame
        def spark_to_csv(df, path):
            tmp_dir = path + "_dir"
            df.select('clean_review', 'label').coalesce(1).write.mode('overwrite').csv(tmp_dir, header=True)
            csv_file = glob.glob(os.path.join(tmp_dir, 'part-*.csv'))[0]
            shutil.move(csv_file, path)
            shutil.rmtree(tmp_dir)
            return path

        train_csv, test_csv = [spark_to_csv(df, f"data/{name}_temp.csv") for df, name in zip([train_df, test_df], ["train", "test"])]

        # Load into pandas
        train_df, test_df = [pd.read_csv(f) for f in [train_csv, test_csv]]
        for f in [train_csv, test_csv]: os.remove(f)

        # Convert to Hugging Face Dataset
        train_ds = Dataset.from_pandas(train_df.rename(columns={'clean_review': 'text'}))
        test_ds = Dataset.from_pandas(test_df.rename(columns={'clean_review': 'text'}))

        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(config['model']['name'], clean_up_tokenization_spaces=True)
        tokenize = lambda batch: tokenizer(batch['text'], padding=True, truncation=True, max_length=512)
        train_ds, test_ds = [ds.map(tokenize, batched=True) for ds in [train_ds, test_ds]]

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model']['name'], num_labels=config['model']['num_labels']
        )

        # Training args
        args = TrainingArguments(
            output_dir=config['training']['output_dir'],
            num_train_epochs=config['training']['epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            per_device_eval_batch_size=config['training']['batch_size'],
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            logging_dir='logs',
            report_to='mlflow'
        )

        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics)

        # Train & log
        with mlflow.start_run():
            trainer.train()
            # Merge model and training params
            params = {**config['model'], **config['training']}
            # Log params in a nested run to avoid conflicts
            with mlflow.start_run(nested=True):
                for key, value in params.items():
                    try:
                        mlflow.log_param(key, value)
                    except mlflow.exceptions.MlflowException as e:
                        print(f"Skipping param {key}: {e}")
            eval_results = trainer.evaluate()
            mlflow.log_metrics(eval_results)
            trainer.save_model(config['training']['output_dir'])
            tokenizer.save_pretrained(config['training']['output_dir'])
            
            # Log model using mlflow.transformers
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                name="model",
                task="text-classification",
                input_example=["This is a sample review to test the model."],
                pip_requirements=["torch", "transformers", "datasets"],
                registered_model_name="SentimentModel"
            )

        print(f"Model trained. Accuracy: {eval_results['eval_accuracy']:.3f}, F1: {eval_results['eval_f1']:.3f}")
        spark.stop()

    except Exception as e:
        print(f"Error in training: {e}")
        raise

if __name__ == "__main__":
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    train_model(config)