import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def _delete_columns(df: pd.DataFrame) -> pd.DataFrame:
    asd