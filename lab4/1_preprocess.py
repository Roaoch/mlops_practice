import pandas as pd

df = pd.read_csv('data/Titanic-Dataset.csv')

df = df[[
    'Pclass',
    'Sex',
    'Age'
]]

df.to_csv('data/Titanic-Dataset.csv', index=False)

print(df.info())