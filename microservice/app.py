from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from models.logistic_regression_model import LogisticRegression
from models.random_forest_model import RandomForestModel

from datetime import datetime

import uvicorn
import pandas as pd


app = FastAPI()
logisticRegressionModel = LogisticRegression()
randomForestModel = RandomForestModel()


class RequestBody(BaseModel):
    year: Optional[int] = None
    buying_events_month_x: int
    buying_events_month_y: int
    all_events_month_x: int
    all_events_month_y: int
    month_sin: float
    month_cos: float
    spent_money_first_month: float
    spent_money_second_month: float
    chosen_model: str


@app.get('/hi')
def hi():
    return 'Hi there :)'


@app.post('/predict')
def predict(body: RequestBody):

    body = body.dict()

    year = body['year']
    chosen_model = body['chosen_model']

    if not year:
        year = 2022

    if year < 2019 or year > 2022:
        return JSONResponse(
            content={'error': 'Invalid year!'},
            status_code=404
        )

    model = None

    if chosen_model == 'LOGISTIC_REGRESSION':
        model = LogisticRegression()
    elif chosen_model == 'RANDOM_FOREST':
        model = RandomForestModel()
    else:
        return JSONResponse(
            content={'error': 'Invalid chosen model!'},
            status_code=404
        )

    del body['chosen_model']
    df = pd.DataFrame({k: [v] for k, v in body.items()})

    prediction_label = model.predict(df)

    json_result = {
        'timestamp': datetime.now().strftime("%m/%d/%Y, %H:%M:%S:%f"),
        'label': prediction_label
    }

    return JSONResponse(
        content=json_result,
        status_code=200
    )


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
