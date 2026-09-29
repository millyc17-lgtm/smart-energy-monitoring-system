import pandas as pd

df = pd.read_csv("data/raw/block_0.csv")

print(df.head())
print(df.shape)
print(df.info())