import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from functions_scripts.load import read_data
from sklearn.base import BaseEstimator, TransformerMixin

class Preparador(TransformerMixin, BaseEstimator):
    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df
        pd.set_option('future.no_silent_downcasting', True)

    def describe_data(self, df: pd.DataFrame | None = None) -> None:
        if df is None:
            columns = self.df.columns
            describe = self.df.describe()

            for c in columns:
                print(f"\n\nColumna: {c}")
                print(f"DataType: {self.df.loc[:,c].dtype}")
                print(f"Unique Vals: {np.unique(self.df.loc[:,c])}")
                if c in describe:
                    print(f"Description:\n{describe.loc[:,c]}")
        else:
            df = pd.DataFrame(df)
            columns = df.columns
            describe = df.describe()

            for c in columns:
                print(f"\n\nColumna: {c}")
                print(f"DataType: {df.loc[:,c].dtype}")
                print(f"Unique Vals: {np.unique(df.loc[:,c])}")
                if c in describe:
                    print(f"Description:\n{describe.loc[:,c]}")
    
    def column_del(self, idxs, df: pd.DataFrame | None = None) -> None:
        """Elimina las columnas indicadas por id o por nombre del dataframe"""
        if type(idxs) is not int and \
            type(idxs) is not str and \
            type(idxs) is not list:
            raise ValueError("[ERROR]: idxs debe ser un entero, string o una lista de los mismos.")

        if df is not None: # No se hace sobre el dataframe self.df
            if type(idxs) is str or (type(idxs) is list and type(idxs[0]) is str):
                df = df.drop(columns=idxs)
            else:
                df = df.drop(columns=df.columns[idxs])
        else:
            if type(idxs) is str or (type(idxs) is list and type(idxs[0]) is str):
                self.df = self.df.drop(columns=idxs)
            else:
                self.df = self.df.drop(columns=self.df.columns[idxs])

    def column_to_numeric(self, idxs, df: pd.DataFrame | None = None) -> None:
        """Pasa la/las columnas indicadas en idxs de str a numerico, pasando valores invalidos a NaN"""
        if type(idxs) is not int and \
            type(idxs) is not str and \
            type(idxs) is not list:
            raise ValueError("[ERROR]: El/los indices deben ser indicados con enteros, strings o una lista de los mismos.")
        
        if type(idxs) is int or type(idxs) is str:
            idxs = [idxs]

        if df is not None: # No se hace sobre el dataframe self.df
            for idx in idxs:
                if type(idx) is int:
                    if idx < 0 or idx >= len(df.columns):
                        continue
                    col_name = df.columns[idx]
                    # errors coerce cambia no numericos ('?') a NaN
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype(np.float32)
                else:
                    if idx not in df.columns:
                        continue
                    # errors coerce cambia no numericos ('?') a NaN
                    df[idx] = pd.to_numeric(df[idx], errors='coerce').astype(np.float32)
        else:
            for idx in idxs:
                if type(idx) is int:
                    if idx < 0 or idx >= len(self.df.columns):
                        continue
                    col_name = self.df.columns[idx]
                    # errors coerce cambia no numericos ('?') a NaN
                    self.df[col_name] = pd.to_numeric(self.df[col_name], errors='coerce').astype(np.float32)
                else:
                    if idx not in self.df.columns:
                        continue
                    # errors coerce cambia no numericos ('?') a NaN
                    self.df[idx] = pd.to_numeric(self.df[idx], errors='coerce').astype(np.float32)
            
    def column_get_unique(self, idxs, df: pd.DataFrame | None = None) -> None:
        """Muestra los distintos valores encontrados en la/las columnas indicadas en idxs"""
        if type(idxs) is not int and \
            type(idxs) is not str and \
            type(idxs) is not list:
            raise ValueError("[ERROR]: El/los indices deben ser indicados con enteros, strings o una lista de los mismos.")
        
        if type(idxs) is int or type(idxs) is str:
            idxs = [idxs]
        
        if df is not None: # No se hace sobre el dataframe self.df
            for idx in idxs:
                if type(idx) is int:
                    if idx < 0 or idx >= len(df.columns):
                        continue
                    print(f"Unique vals in {df.columns[idx]}: {np.unique(df.iloc[:, idx])}")
                else:
                    if idx not in df.columns:
                        continue
                    print(f"Unique vals in {idx}: {np.unique(df.loc[:, idx])}")        
        else:
            for idx in idxs:
                if type(idx) is int:
                    if idx < 0 or idx >= len(self.df.columns):
                        continue
                    print(f"Unique vals in {self.df.columns[idx]}: {np.unique(self.df.iloc[:, idx])}")
                else:
                    if idx not in self.df.columns:
                        continue
                    print(f"Unique vals in {idx}: {np.unique(self.df.loc[:, idx])}")


    def column_binary_categoric_to_numeric(self, idxs, df: pd.DataFrame | None = None) -> None:
        """Pasa las columnas binarias indicadas en idxs a entero, con la siguiente rúbrica:
            - YES/yes -> 1
            - M/m     -> 1
            - NO/no   -> 0
            - F/f     -> 0"""
        if type(idxs) is not int and \
            type(idxs) is not str and \
            type(idxs) is not list:
            raise ValueError("[ERROR]: El/los indices deben ser indicados con enteros, strings o una lista de los mismos.")
        
        if type(idxs) is int or type(idxs) is str:
            idxs = [idxs]

        if df is not None: # No se hace sobre el dataframe self.df
            for idx in idxs:
                col = df.columns[idx] if type(idx) is int else idx
                df[col] = df[col].replace(
                    to_replace={
                        r"(?i)\b(yes|m)\b": 1,
                        r"(?i)\b(no|f)\b": 0
                    },
                    regex=True
                )
                df[col] = pd.to_numeric(df[col], errors='coerce').astype("int8")
        else:
            for idx in idxs:
                col = self.df.columns[idx] if type(idx) is int else idx
                self.df[col] = self.df[col].replace(
                    to_replace={
                        r"(?i)\b(yes|m)\b": 1,
                        r"(?i)\b(no|f)\b": 0
                    },
                    regex=True
                )
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype("int8")

    def column_rename(self, idxs, names=None, df: pd.DataFrame | None = None) -> None:
        """Renombra las columnas indicadas en idxs por los nombres indicados en names"""
        if names is None:
            return
        if type(idxs) is not int and \
            type(idxs) is not str and \
            type(idxs) is not list:
            raise ValueError("[ERROR]: El/los indices deben ser indicados con enteros, strings o una lista de los mismos.")
        if type(idxs) is not str and \
            type(idxs) is not list:
            raise ValueError("[ERROR]: El/los nombres deben ser indicados con strings o una lista de los mismos.")
        if len(idxs) != len(names):
            raise ValueError("[ERROR]: La longitud de idxs y names debe ser la misma.")
        
        if type(names) is str:
            names = [names]
        
        if df is not None: # No se hace sobre el dataframe self.df
            if type(idxs) is int or (type(idxs) is list and type(idxs[0]) is int):
                idxs = [df.columns[idx] for idx in idxs]

            replace_dic = dict([(prev, new) for prev, new in zip(idxs, names)])    
            df = df.rename(columns=replace_dic)
        else:
            if type(idxs) is int or (type(idxs) is list and type(idxs[0]) is int):
                idxs = [self.df.columns[idx] for idx in idxs]

            replace_dic = dict([(prev, new) for prev, new in zip(idxs, names)])    
            self.df = self.df.rename(columns=replace_dic)

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame):
        df_copy = df.copy()
        
        # 1. RENAMING
        idxs = [14, 15]
        self.column_rename(idxs, ["jaundice", "family_pdd"], df_copy)

        # 2. TO NUMERIC
        self.column_to_numeric("age", df_copy)

        # 3. REDUNDANT COLUMNS
        self.column_del(["id", "age_desc"], df_copy)

        # 4. BINARY CATEGORIC -> BINARY NUMERIC
        idxs_categoric_binary_columns = [11, 13, 14, 16, 19]
        self.column_binary_categoric_to_numeric(idxs_categoric_binary_columns, df_copy)
        return df_copy