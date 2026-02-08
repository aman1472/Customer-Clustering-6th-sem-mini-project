import pandas as pd
from pathlib import Path

PROJECT_ROOT =  Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"


df = pd.read_excel(DATA_PATH)

print(df.head(5))
print("Dataset Shape : " ,df.shape)

