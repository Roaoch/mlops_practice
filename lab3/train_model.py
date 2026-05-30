import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv('Titanic-Dataset.csv').drop([
    'PassengerId',
    'Name',
    'Ticket',
    'Cabin'
], axis=1).dropna().drop_duplicates().sample(frac=1)

y = df['Survived'].to_numpy()

X = df.drop([
    'Survived'
], axis=1)

numeric_features = X.select_dtypes(include='number').columns.to_list()
cat_features = X.select_dtypes(include='object').columns.to_list()

pipeline = Pipeline(
    [
        (
            'preprocessor',
            ColumnTransformer(transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OrdinalEncoder(), cat_features)
            ])
        ),
        (
            'predictor',
            GradientBoostingClassifier()
        )
    ]
)

print(pipeline)
print(cross_val_score(
    pipeline,
    X,
    y,
    cv=10,
    scoring='f1'
).mean())

pipeline.fit(X, y)

print(pipeline.named_steps['preprocessor'].named_transformers_['cat'].categories_)

with open('titanic_prediction.pkl', 'wb') as f:
    pickle.dump(pipeline, f)