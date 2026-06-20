from fastapi import FastAPI
from pydantic import BaseModel
from src.model_registry import ModelRegistry

app = FastAPI(
    title="Sentiment Analysis Api",
    Description="Api for Movie Sentiment Analysis Review",
    version="1.0.0",
)

registry = ModelRegistry()

try:
    # Load saved model
    model, metadata = registry.load_model("logistic_regression")
except FileNotFoundError:
    model = None
    metadata = None


class PredictionRequest(BaseModel):
    review: str


@app.get("/")
def heath_check():

    return {"status": "Api up and running...", "model_loaded": model is not None}


@app.post("/predict")
def predict_sentiment(request: PredictionRequest):

    if model is None:
        raise FileNotFoundError(status_code=500, detail="Model not to be found")

    prediction = model.predict([request.review])[0]
    probability = model.predict_proba([request.review])[0][1]

    sentiment = "positive" if prediction == 1 else "negative"

    return {
        "review": request.review,
        "prediction": int(prediction),
        "sentiment": sentiment,
        "probability": round(float(probability), 4),
    }
