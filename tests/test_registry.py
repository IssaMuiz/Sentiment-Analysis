from src.model_registry import ModelRegistry
from sklearn.linear_model import LogisticRegression


def test_save_and_load_model(tmp_path):
    model = LogisticRegression()
    registry = ModelRegistry(model_dir=tmp_path)

    registry.save_model(model, "test_model")

    load_model, _ = registry.load_model("test_model")

    assert load_model is not None
