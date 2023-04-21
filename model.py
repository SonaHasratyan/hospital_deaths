import numpy as np
import pandas as pd

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
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
from matplotlib import pyplot

class Model:
    """
    Includes fit and predict functions, which work as usual as always.
    It is considered, that the input is already preprocessed here.
    """

    def __init__(self, random_state=78, do_validation=True):
        self.threshold = None
        self.y_val = None
        self.X_val = None
        self.y_test = None
        self.do_validation = do_validation
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.features_permutation = None
        self.features_mdi = None
        self.random_state = random_state
        self.model = None

    def fit(self, X_train, y_train):
        if self.do_validation:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.15,
                random_state=self.random_state,
                stratify=y_train,
            )
            self.X_val = X_val
            self.y_val = y_val

        self.X_train = X_train
        self.y_train = y_train

        self.choose_model()
        self.model.fit(self.X_train, self.y_train)
        self.threshold_selection()

    def predict(self, X_test):
        self.X_test = X_test
        y_pred = self.model.predict(self.X_test)

        return y_pred

    def score(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        y_pred = (self.model.predict_proba(self.X_test)[:, 1] >= self.threshold).astype(bool)

        print(f"Train accuracy: {self.model.score(self.X_train, self.y_train)}")
        print(f"Test accuracy: {accuracy_score(self.y_test, y_pred)}")
        print(f"confusion_matrix: {confusion_matrix(self.y_test, y_pred)}")

    def choose_model(self):
        rf = RandomForestClassifier(
            max_depth=5,
            criterion="gini",
            max_leaf_nodes=14,
            random_state=self.random_state,
        )

        knn = KNeighborsClassifier(n_neighbors=8)

        dt = DecisionTreeClassifier(max_depth=5,
            criterion="gini",
            max_leaf_nodes=14)

        self.model = knn

        params = {
            "n_neighbors": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9
            ]
        }

        # Regressor = GridSearchCV(self.model, params, scoring="neg_mean_squared_error", cv=5)
        # Regressor.fit(self.X_train, self.y_train)
        # print(Regressor.best_params_)

    def threshold_selection(self):
        predict_probas = self.model.predict_proba(self.X_val)
        predict_probas = predict_probas[:, 1]

        auc = roc_auc_score(self.y_val, predict_probas)
        print("ROC AUC=%.3f" % auc)

        fpr, tpr, thresholds = roc_curve(self.y_val, predict_probas)
        print(thresholds)

        J = tpr - fpr
        ix = np.argmax(J)
        print("Best Threshold=%f" % thresholds[ix])
        self.threshold = thresholds[ix]
        pyplot.plot(fpr, tpr, marker='.')
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # pyplot.show()
