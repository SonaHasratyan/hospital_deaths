import numpy as np
import pandas as pd

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


class Model:
    """
        Includes fit and predict functions, which work as usual as always.
        It is considered, that the input is already preprocessed here.
    """

    def __init__(self):
        pass

    def fit(self):
        # self.__feature_selection_mdi()
        # self.__feature_selection_permutation()
        # print(
        #     set(self.features_mdi.keys()).intersection(
        #         set(self.features_permutation.keys())
        #     )
        # )
        pass

    def predict(self):
        pass

    def __feature_selection_mdi(self):
        rf = RandomForestClassifier(random_state=self.random_state)
        rf.fit(self.X_train, self.y_train)

        forest_importances = pd.Series(
            rf.feature_importances_, index=self.X_train.columns
        )
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

        print(forest_importances.sort_values())
        fig, ax = plt.subplots()
        forest_importances[forest_importances <= 0.009].plot.bar(
            yerr=std[forest_importances <= 0.009], ax=ax
        )
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        fig.show()

        # todo: make this features v
        self.features_mdi = forest_importances[forest_importances > 0.009]
        print(len(self.features_mdi))

    def __feature_selection_permutation(self):
        rf = RandomForestClassifier(random_state=self.random_state)
        rf.fit(self.X_train, self.y_train)

        permutation = permutation_importance(
            rf,
            self.X_val,
            self.y_val,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=2,
        )
        forest_importances = pd.Series(
            permutation.importances_mean, index=self.X_val.columns
        )
        print(forest_importances.sort_values())
        fig, ax = plt.subplots()
        forest_importances[forest_importances <= 0.0].plot.bar(
            yerr=permutation.importances_std[forest_importances <= 0.0], ax=ax
        )
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        fig.show()

        # todo: make this features v
        self.features_permutation = forest_importances[forest_importances > 0.0]
        print(len(self.features_permutation))

