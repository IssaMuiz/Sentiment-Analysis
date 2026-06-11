from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer

from src.features.text_preprocessing import preprocess_text


def preprocess_batch(X):
    return [preprocess_text(text) for text in X]


def build_pipeline():
    """
    Build a machine learning pipeline that includes text preprocessing,
    TF-IDF vectorization, and a logistic regression classifier.
    Returns:
        Pipeline: A scikit-learn Pipeline object that can be used for
        training and prediction.
    """
    try:

        pipeline = Pipeline(
            steps=[
                (
                    "preprocess_text",
                    FunctionTransformer(preprocess_batch),
                ),
                ("tfidf_vectorizer", TfidfVectorizer(max_features=5000)),
                ("model", LogisticRegression()),
            ]
        )
        print(
            "Pipeline built successfully with text preprocessing, "
            "TF-IDF vectorization, and logistic regression classifier."
        )
        return pipeline
    except Exception as e:
        print(f"Error building pipeline: {e}")
        return None
