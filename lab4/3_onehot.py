import pandas as pd

df = pd.read_csv('data/Titanic-Dataset.csv')

df = pd.get_dummies(df, columns=['Sex'])

df.to_csv('data/Titanic-Dataset.csv', index=False)

print(df.info())