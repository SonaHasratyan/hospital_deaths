import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     mean_squared_error,
#     mean_absolute_error,
#     r2_score,
# )


# import pickle


class Preprocessor:
    """
    All the preprocessing stages are done here - filling nans, scaling, feature extraction etc.
    """

    NANS_THRESHOLD = 60

    def __init__(self, random_state=78):
        self.X_train = None
        self.y_train = None
        self.X = None
        self.selected_features = None
        self.scaler = None

        self.random_state = random_state

    # TODO: KNNImputer instead of median

    def fit(self, X_train, y_train):
        # Just in case checking whether there are any data points with nan labels, if so, remove them
        if y_train.isna().sum() != 0:
            y_train.drop(y_train.isna(), inplace=True)
            X_train.drop(y_train.isna(), inplace=True)

        self.X_train, self.y_train = shuffle(
            X_train, y_train, random_state=self.random_state
        )

        self.__anomaly_detection()

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X_train)
        self.X_train[self.X_train.columns] = self.scaler.transform(
            self.X_train[self.X_train.columns]
        )

        self.__select_features()

    def transform(self, X):
        self.X = X
        # TODO discuss: if any extra column should we drop?

        self.X[self.X.columns] = self.scaler.transform(self.X[self.X.columns])
        self.X = self.__regularize_data(self.X)

        # TODO take .values if needed v
        # self.X = self.X.values

    def __regularize_data(self, X):

        print(f"Number of columns BEFORE dropping: {len(X.columns)}")
        X = X[self.selected_features].copy()

        print(f"Number of columns AFTER dropping: {len(X.columns)}")

        X.fillna(X.median(), inplace=True)
        print(
            f"Are there left any columns with nan values? - {any((X.isna().sum() * 100) / len(X) > 0)}"
        )

        return X

    def __select_features(self):
        """
        removing those columns which include over NANS_THRESHOLD % nans
        filling nans
        setting self.select_features
        """

        nans_percentage = (
            (self.X_train.isna().sum() * 100) / len(self.X_train)
        ).sort_values(ascending=False)
        cols_with_nans_stats = pd.DataFrame(
            {"columns": self.X_train.columns, "nans_percentage": nans_percentage}
        )
        cols_with_nans_stats.sort_values("nans_percentage", inplace=True)

        print(
            f"Number of columns including nans: "
            f'{len(cols_with_nans_stats[cols_with_nans_stats["nans_percentage"] > 0])}'
        )

        features_to_drop = cols_with_nans_stats[
            cols_with_nans_stats["nans_percentage"] > self.NANS_THRESHOLD
        ]["columns"].to_numpy()

        print(
            f"Number of columns with > {self.NANS_THRESHOLD}% nan values: {len(features_to_drop)}"
        )

        X_tmp = self.X_train[self.X_train.columns.difference(features_to_drop)].copy()
        X_tmp.fillna(X_tmp.median(), inplace=True)

        alpha = self.__get_lasso_alpha(X_tmp)

        feature_sel_model = SelectFromModel(
            Lasso(alpha=alpha, random_state=self.random_state)
        )

        feature_sel_model.fit(X_tmp, self.y_train)

        # list of the selected features
        self.selected_features = X_tmp.columns[(feature_sel_model.get_support())]

        # let's print some stats
        print(f"#Total features: {self.X_train.shape[1]}")
        print(f"#Selected features: {len(self.selected_features)}")

    def __anomaly_detection(self):
        # TODO: validate quantiles
        # Identify potential outliers for each column
        outliers = {}
        for col in self.X_train.columns:
            # print(self.X_train[col].describe())
            q1 = self.X_train[col].quantile(0.05)
            q3 = self.X_train[col].quantile(0.95)
            iqr = q3 - q1
            upper_lim = q3 + 1.5 * iqr
            lower_lim = q1 - 1.5 * iqr
            outliers[col] = self.X_train.loc[
                (self.X_train[col] < lower_lim) | (self.X_train[col] > upper_lim), col
            ]
            self.X_train.loc[
                (self.X_train[col] < lower_lim) | (self.X_train[col] > upper_lim), col
            ] = np.nan
            # print(self.X_train[col].describe())
        # print(outliers)

    def __get_lasso_alpha(self, X_tmp, grid_search=False):
        if grid_search:
            from sklearn.model_selection import GridSearchCV

            lasso = Lasso(random_state=self.random_state)
            params = {
                "alpha": [
                    1e-5,
                    1e-4,
                    1e-3,
                    1e-2,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    10,
                    20,
                    30,
                    40,
                    50,
                    100,
                    200,
                    300,
                    400,
                    500,
                ]
            }
            Regressor = GridSearchCV(lasso, params, scoring="neg_mean_squared_error", cv=10)
            Regressor.fit(X_tmp, self.y_train)
            print("best parameter: ", Regressor.best_params_)
            print("best score: ", -Regressor.best_score_)
            return Regressor.best_params_["alpha"]
        else:
            return 0.001


df = pd.read_csv("hospital_deaths_train.csv")
y = df["In-hospital_death"]
X = df.drop("In-hospital_death", axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=78,
    stratify=y,
)

preprocessor = Preprocessor()
preprocessor.fit(X_train, y_train)
preprocessor.transform(X_val)
