from fastapi import FastAPI
from pydantic import BaseModel
from src.model_registry import ModelRegistry

# Initialize the fastapi app
app = FastAPI(
    title="Sentiment Analysis Api",
    Description="Api for Movie Sentiment Analysis Review",
    version="1.0.0",
)

# create an instace of the model_registry class
registry = ModelRegistry()

# Load saved model
try:
    model, metadata = registry.load_model("logistic_regression")
except FileNotFoundError:
    model = None
    metadata = None


# request format
class PredictionRequest(BaseModel):
    review: str


# model status checking route
@app.get("/")
def health_check():

    return {"status": "Api up and running...", "model_loaded": model is not None}


# prediction route
@app.post("/predict")
def predict_sentiment(request: PredictionRequest):

    if model is None:
        raise FileNotFoundError(status_code=500, detail="Model not to be found")

    # model prediction and the probability of the given prediction
    prediction = model.predict([request.review])[0]
    probability = model.predict_proba([request.review])[0][1]

    # sentiment prediction logic
    sentiment = "positive" if prediction == 1 else "negative"

    return {
        "review": request.review,
        "prediction": int(prediction),
        "sentiment": sentiment,
        "probability": round(float(probability), 4),
    }
