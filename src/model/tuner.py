from sklearn.model_selection import GridSearchCV


class TuneModel:
    def __init__(self, pipeline, param_grid):
        self.pipeline = pipeline
        self.param_grid = param_grid

    def tune(self, X_train, y_train):
        grid = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )

        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        print(grid.best_params_)
        print(grid.best_score_)
        return best_estimator
