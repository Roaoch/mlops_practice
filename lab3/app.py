import os
import pickle

import gradio as gr
import pandas as pd
import numpy as np

from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline


PORT = int(os.environ.get('PORT', 8000))


class ESex(Enum):
    Female='female'
    Male='male'


class EEmbarked(Enum):
    Cherbourg='C'
    Queenstown='Q'
    Southampton='S'


class EClass(Enum):
    FirstClass=1
    SecondClass=2
    ThirdClass=3


class PredictIn(BaseModel):
    Pclass: EClass
    Sex: ESex
    Age: int
    SibSp: int
    Parch: int
    Fare: int
    Embarked: EEmbarked


with open('titanic_prediction.pkl', 'rb') as f:
    model: Pipeline = pickle.load(f)


def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    df = pd.DataFrame([
        {
            'Pclass': Pclass,
            'Sex': Sex,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare,
            'Embarked': Embarked 
        }
    ])

    probas = model.predict_proba(df)[0]
    res = np.argmax(probas)

    return res, probas[res]
    

gradio_app = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown(
            [(c.name,  c.value) for c in EClass],
            label='Класс билета',
            value=EClass.FirstClass.value
        ),
        gr.Dropdown(
            [(c.name,  c.value) for c in ESex],
            label='Пол',
            value=ESex.Male.value
        ),
        gr.Number(
            label='Возраст',
            value=22,
            precision=0
        ),
        gr.Number(
            label='Количество братьев и сестер / супругов',
            value=1,
            precision=0
        ),
        gr.Number(
            label='Количество родителей / детей',
            value=1,
            precision=0
        ),
        gr.Number(
            label='Цена билета в $',
            value=30,
            precision=4
        ),
        gr.Dropdown(
            [(c.name,  c.value) for c in EEmbarked],
            label='Порт посадки',
            value=EEmbarked.Queenstown.value
        ),
    ],
    outputs=[
        gr.Number(
            label='Класс'
        ),
        gr.Number(
            label='Вероятность'
        )
    ]
)

app = FastAPI(title='Titanic prediction')

@app.post('/predict')
def predict(body: PredictIn):
    c, prob = predict_survival(**body.model_dump())
    return {
        'class': c,
        'proba': prob
    }

app = gr.mount_gradio_app(app, gradio_app, path='/view')

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host='0.0.0.0',
        port=PORT
    )