import pandas as pd


df  = pd.read_csv("data/raw.csv")
important_columns = ['ram', 'battery_power', 'px_width', 'px_height', 'mobile_wt', 'int_memory','price_range']
df =  df[important_columns]

df.to_csv("mobile_data.csv")