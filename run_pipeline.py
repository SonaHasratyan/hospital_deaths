import pandas as pd
import argparse
from sklearn.model_selection import train_test_split


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
        self.random_state = 78

    def run(self, data_path, inference):
        is_train = False

        if inference == "train":
            is_train = True

        df = pd.read_csv(data_path)

        if is_train:

            y = df["In-hospital_death"]
            X = df.drop("In-hospital_death", axis=1)
            # todo: shuffle
            # todo: the train test separation should come from run file
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y,
            )


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
