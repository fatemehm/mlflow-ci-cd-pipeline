from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import yaml

app = FastAPI()
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
sentiment_pipeline = pipeline('sentiment-analysis', model=config['training']['output_dir'])

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict(review: Review):
    try:
        result = sentiment_pipeline(review.text)[0]
        return {"sentiment": "positive" if result['label'] == 'LABEL_1' else "negative", "score": result['score']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))