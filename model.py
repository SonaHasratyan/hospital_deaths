import numpy as np
import pandas as pd

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
        self.y_test = None
        self.do_validation = do_validation
        self.X_train = None
        self.y_train = None
        self.X_test = None
        #  self.features_permutation = np.array(['ALP_first', 'AST_first', 'Albumin_last', 'BUN_last',
        # 'Bilirubin_first', 'Bilirubin_last', 'Creatinine_first',
        # 'DiasABP_last', 'GCS_highest', 'GCS_last', 'GCS_lowest',
        # 'Glucose_highest', 'Glucose_last', 'HR_highest', 'HR_last',
        # 'Lactate_last', 'MechVentDuration', 'MechVentStartTime',
        # 'NIDiasABP_lowest', 'NIDiasABP_median', 'NIMAP_last', 'PaO2_first',
        # 'SysABP_first', 'Temp_last', 'Temp_lowest', 'Temp_median',
        # 'Weight_last'])
        self.features_permutation = []
        self.features_mdi = None
        self.random_state = random_state
        self.choose_model()

    def fit(self, X_train, y_train):

        # if len(self.features_permutation):
        #     X_train = X_train[self.features_permutation]
        self.threshold_selection(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test):
        if len(self.features_permutation):
            X_test = X_test[self.features_permutation]
        self.X_test = X_test

        return self.model.predict_proba(self.X_test)[:, 1]

    def score(self, X_test, y_test):
        if len(self.features_permutation):
            X_test = X_test[self.features_permutation]

        self.X_test = X_test
        self.y_test = y_test
        y_pred = (self.model.predict_proba(self.X_test)[:, 1] >= self.threshold).astype(
            bool
        )
        cm = confusion_matrix(self.y_test, y_pred)

        print(f"Train accuracy: {self.model.score(self.X_train, self.y_train)}")
        print(f"Test accuracy: {accuracy_score(self.y_test, y_pred)}")
        print(f"confusion_matrix: {confusion_matrix(self.y_test, y_pred)}")
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
        # calculate the specificity
        specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        # print the results
        print('Sensitivity:', sensitivity)
        print('Specificity:', specificity)

    def choose_model(self):
        rf = RandomForestClassifier(
            max_depth=5,
            criterion="entropy",
            max_leaf_nodes=16,
            random_state=self.random_state,
        )

        knn = KNeighborsClassifier(n_neighbors=5, metric="manhattan")

        dt = DecisionTreeClassifier(
            max_features="sqrt",
            criterion="entropy",
            max_depth=6,
            max_leaf_nodes=13,
            random_state=self.random_state,
        )

        logistic = LogisticRegression(
            solver="lbfgs",
            max_iter=1500,
            C=22,
            penalty="l2",
            random_state=self.random_state,
            tol=0.9,
        )

        nb = GaussianNB(var_smoothing=1e-8)

        qda = QuadraticDiscriminantAnalysis(tol=0.001, store_covariance=True)

        svm = SVC(probability=True, degree=1, kernel="poly", C=90)

        self.model = logistic

        do_grid_search = False
        params = {}
        if self.model == logistic:
            params = {
                'penalty': ['l2', 'elasticnet', 'none'],
                # "C": [18, 20, 22, 25],
                "tol": [1e-3, 0.9, 0.1],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                # 'max_iter': [1500],
            }

        if self.model == nb:
            params = {
                "var_smoothing": [1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13],
            }

        if self.model == qda:

            params = {
                "reg_param": [0.0, 0.1, 0.2, 0.3],
                "tol": [1e-3, 1e-4, 1e-5, 0.9, 0.8, 0.5, 0.1],
                "store_covariance": [True, False],
            }

        if self.model in [rf, dt]:
            params = {
                # "criterion": ["gini", "entropy"],
                # "max_depth": [4,5,6,7,8],
                "max_leaf_nodes": [
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    16,
                ],
                # "n_estimators": [80,90,100,110,120]
            }

        if self.model == knn:
            params = {
                "n_neighbors": [4, 5, 6, 7, 8],
                "metric": ["manhattan", "euclidean", "minkowski", "chebyshev"],
            }

        if self.model == svm:
            params = {
                "degree": [
                    1,
                    2,
                    3,
                ],
                # "kernel": ["poly", "rbf", "sigmoid"],
                # "C": [90, 100, 110]
            }

        if do_grid_search and params:
            Regressor = GridSearchCV(
                self.model, params, scoring=make_scorer(roc_auc_score), cv=5
            )
            Regressor.fit(self.X_train, self.y_train)
            print(Regressor.best_params_)

        # if self.model == rf:
        #     self.__feature_selection_permutation()

    def threshold_selection(self, X_train, y_train):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.15,
            random_state=self.random_state,
            stratify=y_train,
        )

        # if len(self.features_permutation):
        #     X_val = X_val[self.features_permutation]

        self.model.fit(X_train, y_train)
        predict_probas = self.model.predict_proba(X_val)
        predict_probas = predict_probas[:, 1]

        auc = roc_auc_score(y_val, predict_probas)
        print("ROC AUC=%.3f" % auc)

        fpr, tpr, thresholds = roc_curve(y_val, predict_probas)
        # print(thresholds)

        J = tpr - fpr
        ix = np.argmax(J)
        print("Best Threshold=%f" % thresholds[ix])
        self.threshold = thresholds[ix]
        pyplot.plot(fpr, tpr, marker=".")
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("True Positive Rate")
        # pyplot.show()

        return self.threshold

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
        # print(forest_importances.sort_values())
        fig, ax = pyplot.subplots()
        forest_importances.plot.bar(yerr=permutation.importances_std, ax=ax)
        ax.set_title("Feature importances using Permutation")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        # fig.show()

        self.features_permutation = forest_importances[forest_importances > 0.0].index
        print(len(self.features_permutation))
        self.X_train = self.X_train[self.features_permutation]
        print(self.features_permutation)
        self.model.fit(self.X_train, self.y_train)
