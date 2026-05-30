import pandas as pd

df = pd.read_csv('data/Titanic-Dataset.csv')

df['Age'] = df['Age'].fillna(df['Age'].mean())

df.to_csv('data/Titanic-Dataset.csv', index=False)

print(df.info())