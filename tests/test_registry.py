from src.model_registry import ModelRegistry
from sklearn.linear_model import LogisticRegression


def test_save_and_load_model():
    model = LogisticRegression()
    registry = ModelRegistry()

    registry.save_model(model, "logistic_regression")

    load_model, _ = registry.load_model("logistic_regression")

    assert load_model is not None
