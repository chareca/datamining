import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from load import read_data

def _01_organize_columns(df: pd.DataFrame) -> pd.DataFrame:

    # renombrado nombre incorrecto 'jundice'->'jaundice'
    df.rename(columns={"jundice": "jaundice"})
    # primera columna ID innecesaria
    df = df.iloc[:,1:]

    idxs_numeric_cols = [11, 18]

    for idx in idxs_numeric_cols:
        # errors coerce cambia no numericos ('?') a NaN
        df.iloc[:, idx] = pd.to_numeric(df.iloc[:, idx], errors='coerce')


    # columna age_desc -> 

    return df

def _02_categoric_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    pass

df = _01_organize_columns(read_data())

print(df.head())
print(df.describe())

for c in df.columns:
    print(c)
    print(np.unique(df.loc[:,c]))