from src.pipeline.build_pipeline import build_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TrainModel:
    def __init__(self):
        self.pipeline = build_pipeline()

    def fit(self, X_train, y_train):
        """
        Fit the pipeline to the training data.
        """
        self.pipeline.fit(X_train, y_train)
        return self.pipeline

    def predict(self, X_test):
        """
        Predict the target variable for the test data.
        """
        return self.pipeline.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance on the test data.
        """
        y_pred = self.predict(X_test)
        evaluation_results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
        }
        return evaluation_results
