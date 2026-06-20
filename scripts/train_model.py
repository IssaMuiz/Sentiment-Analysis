from src.model_registry import ModelRegistry
from src.model.trainer import TrainModel
from scripts.prepare_data import X_train, y_train, X_val, y_val
from sklearn.linear_model import LogisticRegression

# an instance of the modelregistry class
registry = ModelRegistry()

# Iniitializing and training the final best model choosen
model = LogisticRegression()
model_name = "logistic_regression"
train_model = TrainModel(model)

# fit the model
model = train_model.fit(X_train, y_train)

# save the model
registry.save_model(
    model,
    model_name,
    metadata={
        "project_name": "Sentiment Analysis Model",
        "model_type": model_name,
        "version": "1",
        "metrics": train_model.evaluate(X_val, y_val, X_train, y_train),
        "dataset": {"name": "IMDB Movie Reviews", "size": 49582},
        "preprocessing": {
            "vectorizer": "TfidfVectorizer",
            "max_features": 5000,
            "ngram_range": (1, 2),
        },
        "trained_at": "2026-06-19",
    },
)
