import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from load import read_data

def _01_organize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # primera columna ID innecesaria
    # df = df.iloc[:,1:]

    print(df.columns)
    # renombrado nombre incorrecto 'jundice'->'jaundice'
    df.rename(columns={"jundice": "jaundice"})

    idxs_numeric_cols = [11, 18]

    for idx in idxs_numeric_cols:
        # errors coerce cambia no numericos ('?') a NaN
        df.loc[:, idx] = pd.to_numeric(df.loc[:, idx], errors='coerce')

    return df

def _02_categoric_(df: pd.DataFrame) -> pd.DataFrame:
    pass

df = _01_organize_columns(read_data())

print(df.head())
print(df.describe())

for c in df.columns:
    print(c)
    print(np.unique(df.loc[:,c]))