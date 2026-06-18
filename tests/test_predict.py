from src.model_registry import ModelRegistry


def test_predict():

    registry = ModelRegistry()
    model, metadata = registry.load_model("logistic_regression")
    result = model.predict(["I love this movie"])

    assert result is not None
