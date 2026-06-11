import os
import joblib
import json


class ModelRegistry:
    """
    A class to manage the saving and loading of machine learning models and their metadata.
    Attributes:
        model_dir (str): The directory where models and metadata will be stored.
    """

    def __init__(self, model_dir="artifacts/models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def save_model(self, model, model_name, metadata=None):
        """
        Save a machine learning model and its metadata to the registry.
        Args:
            model: The trained model to be saved.
            model_name (str): The name of the model.
            metadata (dict, optional): Additional metadata for the model.
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)

        if metadata is not None:
            metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)
        print(f"Model '{model_name}' saved successfully with metadata: {metadata}")

    def load_model(self, model_name):
        """
        Load a machine learning model and its metadata from the registry.
        Args:
            model_name (str): The name of the model to be loaded.
        Returns:
            tuple: The loaded model and its metadata.
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        model = joblib.load(model_path)

        metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = None

        return model, metadata
