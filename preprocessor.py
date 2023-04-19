import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# import pickle


class Preprocessor:
    """
    All the preprocessing stages are done here - filling nans, scaling, feature extraction etc.
    """

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

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )

        self.X_train, self.y_train = self.__fill_nans(
            self.X_train, self.y_train, is_train=True
        )
        self.X_val, self.y_val = self.__fill_nans(self.X_val, self.y_val)

        self.X_train = self.__scale(self.X_train, is_train=True)
        self.X_val = self.__scale(self.X_val)

    def transform(self):
        X = self.X_train
        y = self.y_train
        return X, y

    def find_strings(self, lst):
        # finds features of names like 'analyze_name_first', 'analyze_name_last ',
        # 'analyze_name_highest','analyze_name_lowest', 'analyze_name_median' make labels,

        strings_dict = {}

        for string in lst:
            words = string.split('_')

            if words[0] in strings_dict:
                strings_dict[words[0]].append(string)
            else:
                strings_dict[words[0]] = [string]

        output_strings = []

        for strings_list in strings_dict.values():
            for string in strings_list:
                if string.endswith('_last'):
                    output_strings.append(string)
                elif string.endswith('_first'):
                    output_strings.append(string)
                elif string.endswith('_lowest'):
                    output_strings.append(string)
                elif string.endswith('_highest'):
                    output_strings.append(string)
                elif string.endswith('_median'):
                    output_strings.append(string)
        return output_strings

    def __fill_nans(self, X, y, is_train=False):
        """
        removing data point which have nan labels
        removing those columns which include over 70% nans
        filling nans
        setting X, y
        """

        # TODO: discuss whether we should drop recordid or not
        X = X.drop("recordid", axis=1)

        if is_train:
            print("-----TRAIN CASE-----")
            nans_percentage = (X.isna().sum() * 100) / len(X)
            columns_with_nans_statistics = pd.DataFrame(
                {"columns": X.columns, "nans_percentage": nans_percentage})

            columns_with_nans_statistics.sort_values("nans_percentage", inplace=True)

            print(
                f'Number of columns including nans: {len(columns_with_nans_statistics[columns_with_nans_statistics["nans_percentage"] > 0])}'
            )
            # print(columns_with_nans_statistics[columns_with_nans_statistics["nans_percentage"] > 70]["columns"])

            self.features_to_drop = columns_with_nans_statistics[
                columns_with_nans_statistics["nans_percentage"] > 70
                ]["columns"]
        else:
            print("-----VALIDATION/TEST CASE-----")

        print(
            f"Number of columns BEFORE dropping columns with > 70% nan values: {len(X.columns)}"
        )

        X = X.drop(
            self.features_to_drop,
            axis=1,
        )
        print(
            f"Number of columns AFTER dropping columns with > 70% nan values: {len(X.columns)}"
        )

        # X.fillna(X.mean(), inplace=True)
        print(
            f"Are there left any columns with nan values? - {any((X.isna().sum() * 100) / len(X) > 0)}"
        )

        # supposed that weight and height are not correlated with any of other features, so fill Nan
        # values of them them with random values from corresponding ranges

        X['Weight'] = X['Weight'].fillna(pd.Series(np.random.randint(50, 100, size=len(X))))
        # X['Weight_last'] = X['Weight_last'].fillna(pd.Series(np.random.randint(50, 100, size=len(X))))

        X['Height'] = X['Height'].replace(X['Height'][X['Height'] > 250].index, np.nan)
        X['Height'] = X['Height'].fillna(pd.Series(np.random.randint(100, 210, size=len(X))))
        # a few Nan values ...
        X['Gender'] = X['Gender'].apply(lambda x: np.random.choice([1, 0], p=[0.5, 0.5]) if pd.isnull(x) else x)
        # and for them
        X['MechVentStartTime'] = X['MechVentStartTime'].fillna(
            pd.Series(np.random.randint(X['MechVentStartTime'].min(), X['MechVentStartTime'].max(), size=len(X))))
        X['MechVentDuration'] = X['MechVentDuration'].fillna(
            pd.Series(np.random.randint(X['MechVentDuration'].min(), X['MechVentDuration'].max(), size=len(X))))
        X['MechVentLast8Hour'] = X['MechVentLast8Hour'].fillna(
            pd.Series(np.random.randint(X['MechVentLast8Hour'].min(), X['MechVentLast8Hour'].max(), size=len(X))))
        X['UrineOutputSum'] = X['UrineOutputSum'].fillna(
            pd.Series(np.random.randint(X['UrineOutputSum'].min(), X['UrineOutputSum'].max(), size=len(X))))

        # for features of names like 'analyze_name_first', 'analyze_name_last ',
        # 'analyze_name_highest','analyze_name_lowest', 'analyze_name_median' make labels,
        # and fill them with mean values of the same label
        labels = ['CCU', 'CSRU',
                  'SICU']  # these features have binar values and no Nans, and are characteristics of care kind
        columns_list = X.columns.tolist()
        proc_data = pd.DataFrame()  # proccessed dataframe here
        # list of names of those features
        features = self.find_strings(columns_list)
        for feature in features:

            full_rows = X.loc[X[feature].isnull() == False, feature]
            missing_rows = X.loc[X[feature].isnull() == True, feature]

            full_rows.name = feature
            missing_rows.name = feature
            indexes = []
            values = []
            for ind in X['CCU'].index:
                indexes.append(ind)
                label_list = [X['CCU'][ind], X['CSRU'][ind], X['SICU'][ind]]
                count_0 = 0
                count_1 = 0
                target_element = [0, 1]

                for element in label_list:
                    if element == target_element[0]:
                        count_0 += 1
                    elif element == target_element[1]:
                        count_1 += 1
                values.append(0 if count_0 > count_1 else 1)
            label = pd.Series(values, index=indexes)  # new Series object including weighted labels of

            for j in missing_rows.index:
                if label[j] == 0:

                    selected_rows = full_rows.loc[label == 0]
                    missing_rows[j] = selected_rows.mean()
                else:
                    selected_rows_ = full_rows.loc[label == 1]
                    missing_rows[j] = selected_rows_.mean()

            D = pd.concat([full_rows, missing_rows])  # change name in class X = df there
            D = D.sort_index()
            proc_data = pd.concat([proc_data, D.to_frame()], axis=1)
        unproc_data = pd.concat(
            [X['Age'].to_frame(), X['Height'].to_frame(), X['Weight'].to_frame(), X['Gender'].to_frame(),
             X['CCU'].to_frame(), X['CSRU'].to_frame(), X['SICU'].to_frame(), X['MechVentStartTime'].to_frame(),
             X['MechVentDuration'].to_frame(), X['MechVentLast8Hour'].to_frame(), X['UrineOutputSum'].to_frame()],
            axis=1)
        data = pd.concat([proc_data, unproc_data], axis=1)
        # X = X.values
        # y = y.values

        return X, y data  ######

    def __scale(self, X, is_train=False):
        if is_train:
            self.scaler = MinMaxScaler()
            self.scaler.fit(X)

        X = self.scaler.transform(X)

        return X


# TODO: close this v
Preprocessor().transform()