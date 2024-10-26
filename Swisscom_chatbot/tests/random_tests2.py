import pandas as pd

df = pd.read_csv("/teamspace/studios/this_studio/resumed.csv")

df = df.head(50)

df.to_csv('resumed_reduced.csv')