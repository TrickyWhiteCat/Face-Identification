# CSV to XLXS
import pandas as pd
filepath = r"C:\Users\nmttu\Downloads\sales2019_1.csv"

df = pd.read_csv(filepath)
df.to_excel("output.xlsx", index=False)