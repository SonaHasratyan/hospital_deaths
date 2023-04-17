import numpy as np
import pandas as pd


class Preprocessor:
    """
    All the preprocessing stages are done here - filling nans, scaling, feature extraction etc.
    """

    def __init__(self):
        self.X = None
        self.y = None

    def fit(self):
        df = pd.read_csv("hospital_deaths_train.csv")

        # Just in case checking whether there are any data points with null labels, if so, remove them
        if df["In-hospital_death"].isna().sum() != 0:
            df.dropna(subset=["In-hospital_death"], inplace=True)

        self.y = df["In-hospital_death"]

        df = df.drop("In-hospital_death", axis=1)

        nans_percentage = (df.isna().sum() * 100) / len(df)
        columns_with_nans_statistics = pd.DataFrame(
            {"columns": df.columns, "nans_percentage": nans_percentage}
        )
        columns_with_nans_statistics.sort_values("nans_percentage", inplace=True)

        print(
            f'Number of columns including null: {len(columns_with_nans_statistics[columns_with_nans_statistics["nans_percentage"] > 0])}'
        )
        # print(columns_with_nans_statistics[columns_with_nans_statistics["nans_percentage"] > 70]["columns"])

        print(
            f"Number of columns BEFORE dropping columns with > 70% null values: {len(df.columns)}"
        )
        df = df.drop(
            columns_with_nans_statistics[
                columns_with_nans_statistics["nans_percentage"] > 70
            ]["columns"],
            axis=1,
        )
        print(
            f"Number of columns AFTER dropping columns with > 70% null values: {len(df.columns)}"
        )

        df.fillna(df.mean(), inplace=True)
        print(
            f"Are there left any columns with null values? - {any((df.isna().sum() * 100) / len(df) > 0)}"
        )

        self.X = df.values
        self.y = self.y.values

    def transform(self):
        pass


# TODO: close this v
Preprocessor().fit()
