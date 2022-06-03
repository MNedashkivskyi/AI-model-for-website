import os
import pandas
import joblib
import pandas as pd

BASE_DIR = 'C:\\Users\\nedma\\PycharmProjects\\IUM'


class RandomForestModel:

    def __init__(self):

        os.chdir(BASE_DIR + f"{os.sep}trained_models{os.sep}")
        self.restoring_filename = 'random_forest_model.pickle'
        try:
            self.model = pd.read_pickle(self.restoring_filename)
        except:
            print("Can't restore a model")

    def predict(self, single_item):
        return float(self.model.predict(single_item)[0])
