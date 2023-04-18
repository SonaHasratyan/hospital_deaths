# import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

        X = X.values
        y = y.values

        return X, y

    def __set_features_to_drop(self, X):
        nans_percentage = (X.isna().sum() * 100) / len(X)
        columns_with_nans_statistics = pd.DataFrame(
            {"columns": X.columns, "nans_percentage": nans_percentage}
        )
        columns_with_nans_statistics.sort_values("nans_percentage", inplace=True)

        print(
            f'Number of columns including nans: '
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


preprocessor = Preprocessor()
preprocessor.fit()
preprocessor.transform()
