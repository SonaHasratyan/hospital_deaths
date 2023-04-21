import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from preprocessor import Preprocessor
from model import Model
import json


"""
    This file can have 2 arguments.
    1.  --data_path: absolute path to a train or test dataset.
    2.  --inference: whether to run model in a training mode or in testing mode. 
        a.  training mode (by default - If the argument is not given)
        b.  testing mode (pass manually as an argument) -  When running in testing mode, outputs are saved in 
            predictions.json file, which has 2 keys:
            1)  predict_probas: predicted probabilities for each test datapoint
            2)  threshold: the best threshold recommended to take
                (all probabilities higher than threshold would be converted to 1, otherwise to 0).
"""


class Pipeline:
    """
    Has a run method.
    """

    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.threshold = 0.5
        self.random_state = 78

    def run(self, data_path, inference):
        df = pd.read_csv(data_path)

        if inference == "train":
            y = df["In-hospital_death"]
            X = df.drop("In-hospital_death", axis=1)
            X, y = shuffle(X, y, random_state=self.random_state)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y,
            )

            # todo: close do_validation
            self.preprocessor = Preprocessor(
                random_state=self.random_state, do_validation=False
            )
            self.preprocessor.fit(X_train, y_train)
            X_train = self.preprocessor.transform(X_train)  # .to_numpy()
            y_train = y_train  # .to_numpy()
            X_test = self.preprocessor.transform(X_test)  # .to_numpy()
            self.model = Model(random_state=self.random_state, do_validation=False)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.model.score(X_test, y_test)

        else:
            # todo: self.threshold - get from validation
            #       maybe auc/ruc for the threshold

            X_test = self.preprocessor.transform(df)  # .to_numpy()
            y_pred = self.model.predict(X_test)

            # Create a dictionary with the data to save
            data = {"predict_probas": y_pred, "threshold": self.threshold}

            # Open the JSON file in write mode and write the data to it
            with open("predictions.json", "w") as outfile:
                json.dump(data, outfile)

            # Load the existing data from the file
            with open("predictions.json", "r") as infile:
                data = json.load(infile)

            print(data)

            # if predict_probas > threshold:
            #     predict_probas = 1
            # else:
            #     predict_probas = 0


# if called for testing, the class would not be fitted. You need to handle this somehow, so testing works properly.
# For example, consider saving trainer and preprocessor after being fit and send us also saved versions of them after
# your final train, so your final result can be repeated without training.

# if run_pipeline.py is run without --inference argument, data in --data_path will include both features and a label,
# if it is run with --inference argument it will not include the label. However, we would also like to test the model
# we fit during the check, so you need to also save model and preprocessor in training mode.


parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--inference", type=str, required=False, default="train")

args = parser.parse_args()

pipeline = Pipeline()
pipeline.run(args.data_path, args.inference)
