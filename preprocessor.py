import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
from sklearn.ensemble import RandomForestClassifier

# import pickle


class Preprocessor:
    """
    All the preprocessing stages are done here - filling nans, scaling, feature extraction etc.
    """

    NANS_THRESHOLD = 60

    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.features_to_drop = None
        self.scaler = None
        self.features = None
        self.random_state = 78

    def fit(self):
        df = pd.read_csv("hospital_deaths_train.csv")

        # Just in case checking whether there are any data points with nan labels, if so, remove them
        if df["In-hospital_death"].isna().sum() != 0:
            df.dropna(subset=["In-hospital_death"], inplace=True)

        self.y = df["In-hospital_death"]
        self.X = df.drop("In-hospital_death", axis=1)

        # todo: the train test separation should come from run file
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )

        self.__set_features_to_drop(self.X_train)

    def transform(self):
        self.X_train, self.y_train = self.__regularize_data(self.X_train, self.y_train)
        self.X_val, self.y_val = self.__regularize_data(self.X_val, self.y_val)

        self.__feature_selection_mdi()
        self.__feature_selection_permutation()
        print(
            set(self.features_mdi.keys()).intersection(
                set(self.features_permutation.keys())
            )
        )

        self.X_train, self.y_train = self.X_train.values, self.y_train.values
        self.X_val, self.y_val = self.X_val.values, self.y_val.values

        self.__anomaly_detection(self.X_train)

        # scaler can't be set in fit method cos' we are dropping some features in transform set
        self.__set_scaler(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)

    def __regularize_data(self, X, y):
        """
        removing data points which have nan labels
        removing those columns which include over NANS_THRESHOLD % nans
        filling nans
        setting X, y
        """

        print(
            f"Number of columns BEFORE dropping columns with > {self.NANS_THRESHOLD}% nan values: {len(X.columns)}"
        )

        # TODO: discuss whether we should drop recordid or not
        X = X.drop("recordid", axis=1)

        X = X.drop(
            self.features_to_drop,
            axis=1,
        )
        print(
            f"Number of columns AFTER dropping columns with > {self.NANS_THRESHOLD}% nan values: {len(X.columns)}"
        )

        X.fillna(X.mean(), inplace=True)
        print(
            f"Are there left any columns with nan values? - {any((X.isna().sum() * 100) / len(X) > 0)}"
        )

        return X, y

    def __set_features_to_drop(self, X):
        nans_percentage = (X.isna().sum() * 100) / len(X)
        columns_with_nans_statistics = pd.DataFrame(
            {"columns": X.columns, "nans_percentage": nans_percentage}
        )
        columns_with_nans_statistics.sort_values("nans_percentage", inplace=True)

        print(
            f"Number of columns including nans: "
            f'{len(columns_with_nans_statistics[columns_with_nans_statistics["nans_percentage"] > 0])}'
        )
        # print(
        #     columns_with_nans_statistics[
        #         columns_with_nans_statistics["nans_percentage"] > self.NANS_THRESHOLD
        #     ]["columns"]
        # )

        self.features_to_drop = columns_with_nans_statistics[
            columns_with_nans_statistics["nans_percentage"] > self.NANS_THRESHOLD
        ]["columns"]

    def __set_scaler(self, X):
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)

    def __anomaly_detection(self, X_train):
        # TODO
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


preprocessor = Preprocessor()
preprocessor.fit()
preprocessor.transform()
