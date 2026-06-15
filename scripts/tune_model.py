from src.pipeline.build_pipeline import build_pipeline
from src.model.tuner import TuneModel
from sklearn.linear_model import LogisticRegression

pipeline = build_pipeline(LogisticRegression(max_iter=1000, solver="liblinear"))

param_grid = {
    "classifier__C": [0.1, 0.5, 1, 2],
    "classifier__penalty": ["l1", "l2"],
    "tfidf_vectorizer__ngram_range": [(1, 1), (1, 2)],
    "tfidf_vectorizer__max_features": [5000, 1000],
}


logistic_reg_tuner = TuneModel(pipeline, param_grid)
