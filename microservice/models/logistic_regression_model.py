
import os
import joblib
import numpy as np

BASE_DIR = 'C:\\Users\\nedma\\PycharmProjects\\IUM'


class LogisticRegression:

    def __init__(self):

        os.chdir(BASE_DIR + f"{os.sep}trained_models{os.sep}")
        self.restoring_filename = 'logistic_regression_model.pickle'
        try:
            self.model = joblib.load(self.restoring_filename)
        except:
            print("Can't restore a model")

    def predict(self, single_item):
        return float(self.model.predict(single_item)[0])


