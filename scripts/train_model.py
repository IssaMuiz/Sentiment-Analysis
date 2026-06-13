from src.model_registry import ModelRegistry
from src.model.train import TrainModel
from scripts.prepare_data import X_train, y_train, X_test, y_test

train_model = TrainModel()
registry = ModelRegistry()

model = train_model.fit(X_train, y_train)

registry.save_model(
    model,
    "RandomForest",
    metadata={
        "model_name": "Sentiment Analysis Model",
        "model_type": "Random Forest",
        "version": "1",
        "metrics": train_model.evaluate(X_test, y_test, X_train, y_train),
        "dataset": {"name": "IMDB Movie Reviews", "size": 49582},
        "preprocessing": {
            "vectorizer": "TfidfVectorizer",
            "max_features": 5000,
        },
        "trained_at": "2026-06-11",
    },
)
