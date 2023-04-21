import numpy as np
import pandas as pd

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import BaggingClassifier


class Model:
    """
    Includes fit and predict functions, which work as usual as always.
    It is considered, that the input is already preprocessed here.
    """

    def __init__(self, random_state=78, do_validation=False):
        self.do_validation = do_validation
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.features_permutation = None
        self.features_mdi = None
        self.random_state = random_state
        self.model = None
        # todo: delete v
        # self.selected_features = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.model = self.__choose_model()
        self.model.fit(self.X_train, self.y_train)

        # self.__feature_selection_mdi()
        # self.__feature_selection_permutation()
        # self.selected_features = self.features_mdi.keys()
        # self.X_train = self.X_train[self.selected_features]

        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        # self.X_test = X_test[self.selected_features]
        self.X_test = X_test
        y_pred = self.model.predict(self.X_test)

        return y_pred

    def score(self, X_test, y_test):
        # X_test = X_test[self.selected_features]
        y_pred = self.predict(X_test)
        self.X_test = X_test
        self.y_test = y_test

        print(f"Train accuracy: {self.model.score(self.X_train, self.y_train)}")
        print(f"Test accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"confusion_matrix: {confusion_matrix(y_test, y_pred)}")
        print(
            f"mean_squared_error: {mean_squared_error(y_test, y_pred, squared=False)}"
        )
        print(f"mean_absolute_error: {mean_absolute_error(y_test, y_pred)}")
        print(f"r2_score: {r2_score(y_test, y_pred)}")

    def __feature_selection_mdi(self):
        forest_importances = pd.Series(
            self.model.feature_importances_, index=self.X_train.columns
        )
        std = np.std(
            [tree.feature_importances_ for tree in self.model.estimators_], axis=0
        )

        print(forest_importances.sort_values())
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        fig.show()

        self.features_mdi = forest_importances[forest_importances > 0.008]
        print(len(self.features_mdi))

    def __feature_selection_permutation(self):
        permutation = permutation_importance(
            self.model,
            self.X_train,
            self.y_train,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=2,
        )
        forest_importances = pd.Series(
            permutation.importances_mean, index=self.X_train.columns
        )
        print(forest_importances.sort_values())
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=permutation.importances_std, ax=ax)
        ax.set_title("Feature importances using Permutation")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        fig.show()

        self.features_permutation = forest_importances[forest_importances > 0.0]
        print(len(self.features_permutation))

    def choose_model(self, X_train, y_train, X_test, y_test):
        rf = RandomForestClassifier(
            max_depth=5,
            criterion="gini",
            max_leaf_nodes=14,
            random_state=self.random_state,
        )
        _model = rf
        bg = BaggingClassifier(_model, n_estimators=20, oob_score=True)
        bg.fit(X_train, y_train)
        y_pred = bg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy:", accuracy)
        print('Training accuracy: ', bg.score(X_train, y_train))
        print('Test accuracy: ', bg.score(X_test, y_test))
        print("OOB Score: ", bg.oob_score_)

        models = [
            KNeighborsClassifier(),
            DecisionTreeClassifier(),
            LogisticRegression(),
        ]

        for model in models:
            if model == KNeighborsClassifier():
                param_grid = {"n_neighbors": range(1, 10)}
            elif model == DecisionTreeClassifier():
                param_grid = {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [4, 5, 6, 7, 8, 10, 12, 13, 14, 15],
                }
            elif model == LogisticRegression():
                param_grid = {"C": [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 4, 5]}

            _model = GridSearchCV(model, param_grid=param_grid, cv=5)
            _model.fit(X_train, y_train)
            bg = BaggingClassifier(_model, n_estimators=20, oob_score=True)
            bg.fit(X_train, y_train)
            y_pred = bg.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(
                "Best hyperparameters for model {}: {}".format(
                    model.__class__.__name__, _model.best_params_
                )
            )
            print("Accuracy:", accuracy)

            print('Training accuracy: ', bg.score(X_train, y_train))
            print('Test accuracy: ', bg.score(X_test, y_test))
            print("OOB Score: ", bg.oob_score_)

        return rf
