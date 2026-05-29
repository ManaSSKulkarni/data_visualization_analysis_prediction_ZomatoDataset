import pandas as pd

df = pd.read_csv("zomato.csv")

df_small = df.sample(5000, random_state=42)

df_small.to_csv("zomato_small.csv", index=False)