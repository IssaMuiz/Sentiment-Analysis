from src.pipeline.build_pipeline import build_pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class TrainModel:
    def __init__(self, models):
        self.pipeline = build_pipeline(models)

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

    def evaluate(self, X_test, y_test, X_train, y_train):
        """
        Evaluate the model's performance on the test data.
        """
        y_pred = self.predict(X_test)
        train_score = self.pipeline.score(X_train, y_train)

        # ROC-AUC score safe handling for different classifier
        if hasattr(self.pipeline, "predict_proba"):
            y_score = self.pipeline.predict_proba(X_test)[:, 1]

        elif hasattr(self.pipeline, "decision_function"):
            y_score = self.pipeline.decision_function(X_test)

        else:
            y_score = y_pred  # fallback

        evaluation_results = {
            "train_score": train_score,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, y_score, average="weighted"),
        }
        return evaluation_results
