from src.model_registry import ModelRegistry
from src.model.trainer import TrainModel
from scripts.prepare_data import X_train, y_train, X_val, y_val
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

models = {
    "logistic_regression": LogisticRegression(),
    "linear_svc": LinearSVC(),
    "multinomial_nb": MultinomialNB(),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
}


registry = ModelRegistry()

for model_name, model in models.items():

    train_model = TrainModel(model)

    model = train_model.fit(X_train, y_train)

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
            "trained_at": "2026-06-11",
        },
    )
