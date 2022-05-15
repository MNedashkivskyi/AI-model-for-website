import uvicorn
import json
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import random
from datetime import datetime

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


class BasicModel:
    def __init__(self):
        self.model_file_name = 'BasicModel.pkl'  # name of file
        try:
            self.model = joblib.load(self.model_file_name)
        except Exception as _:
            self.model = self._train_a_model()
            joblib.dump(self.model, self.model_file_name)

    def _train_a_model(self):
        # An example how it has to be looked

        linear_model = LinearRegression()
        # linear_model.fit(X, y)
        #
        # scores_linear = cross_val_score(linear_model, X=X, y=y, scoring='neg_mmean_absolute_error', cv=5)

        return linear_model

    def predict_species(self, data_features):
        prediction = self.model.predict(data_features)
        return prediction[0]


class Model:
    def __init__(self):
        self.model_file_name = 'Model.pkl'
        try:
            self.model = joblib.load(self.model_file_name)
        except Exception as _:
            self.model = self._train_a_model()
            joblib.dump(self.model, self.model_file_name)

    def _train_a_model(self):
        # An example how it has to be looked

        svr_model = SVR(kernel='rbf', C=10000)
        # svr_model.fit(X, y)

        # scores_svr = cross_val_score(...)

        return svr_model

    def predict_species(self, data_features):
        prediction = self.model.predict(data_features)
        return prediction[0]


# Here will be created data_to_train, X and y datasets

app = FastAPI()
model = Model()
basicModel = BasicModel()


@app.get('/predict')
def predict_species(model_type=None):
    # data_features
    data_features = None
    save_to_file = False
    if model_type is None:
        save_to_file = True
        if random.randint(0, 1) == 0:
            model_type = 'BaseModel'
        else:
            model_type = 'SecondModel'
    if model_type == 'BaseModel':
        prediction = model.predict_species(data_features)
    else:
        prediction = basicModel.predict_species(data_features)

    result = {
        'prediction_time': datetime.now().strftime("%m/%d/%Y %H:%M:%S:%f"),
        #...
        'model_type': model_type,
        'prediction': prediction
    }

    if save_to_file:
        file_object = open('logs/predictions.json', 'a')
        json.dump(result, file_object)
        file_object.write('\n')
        file_object.close()
    return result


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

