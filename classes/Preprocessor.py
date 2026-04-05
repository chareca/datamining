import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from functions_scripts.load import read_data

class Preprocessor():
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def describe_data(self) -> None:
        columns = self.df.columns
        describe = self.df.describe()

        for c in columns:
            print(f"\n\nColumna: {c}")
            print(f"DataType: {self.df.loc[:,c].dtype}")
            print(f"Unique Vals: {np.unique(self.df.loc[:,c])}")
            if c in describe:
                print(f"Description:\n{describe.loc[:,c]}")
    
    def column_del(self, idxs) -> None:
        """Elimina las columnas indicadas por id o por nombre del dataframe"""
        if type(idxs) is not int and \
            type(idxs) is not str and \
            type(idxs) is not list:
            raise ValueError("[ERROR]: idxs debe ser un entero, string o una lista de los mismos.")

        if type(idxs) is str or (type(idxs) is list and type(idxs[0]) is str):
            self.df = self.df.drop(columns=idxs)
        else:
            self.df = self.df.drop(columns=self.df.columns[idxs])

    def column_to_numeric(self, idxs) -> None:
        """Pasa la/las columnas indicadas en idxs de str a numerico, pasando valores invalidos a NaN"""
        if type(idxs) is not int and \
            type(idxs) is not str and \
            type(idxs) is not list:
            raise ValueError("[ERROR]: El/los indices deben ser indicados con enteros, strings o una lista de los mismos.")
        
        if type(idxs) is int or type(idxs) is str:
            idxs = [idxs]

        for idx in idxs:
            if type(idx) is int:
                if idx < 0 or idx >= len(self.df.columns):
                    continue
                # errors coerce cambia no numericos ('?') a NaN
                self.df.iloc[:, idx] = pd.to_numeric(self.df.iloc[:, idx], errors='coerce')
            else:
                if idx not in self.df.columns:
                    continue
                # errors coerce cambia no numericos ('?') a NaN
                self.df.loc[:, idx] = pd.to_numeric(self.df.loc[:, idx], errors='coerce')
            
    def column_get_unique(self, idxs) -> None:
        """Muestra los distintos valores encontrados en la/las columnas indicadas en idxs"""
        if type(idxs) is not int and \
            type(idxs) is not str and \
            type(idxs) is not list:
            raise ValueError("[ERROR]: El/los indices deben ser indicados con enteros, strings o una lista de los mismos.")
        
        if type(idxs) is int or type(idxs) is str:
            idxs = [idxs]
        
        for idx in idxs:
            if type(idx) is int:
                if idx < 0 or idx >= len(self.df.columns):
                    continue
                print(f"Unique vals in {self.df.columns[idx]}: {np.unique(self.df.iloc[:, idx])}")
            else:
                if idx not in self.df.columns:
                    continue
                print(f"Unique vals in {idx}: {np.unique(self.df.loc[:, idx])}")


    def column_binary_categoric_to_numeric(self, idxs) -> None:
        """Pasa las columnas binarias indicadas en idxs a entero, con la siguiente rúbrica:
            - YES/yes -> 1
            - M/m     -> 1
            - NO/no   -> 0
            - F/f     -> 0"""
        if type(idxs) is not int and \
            type(idxs) is not str and \
            type(idxs) is not list:
            raise ValueError("[ERROR]: El/los indices deben ser indicados con enteros, strings o una lista de los mismos.")
        
        if type(idxs) is int or (type(idxs) is list and type(idxs[0]) is int):
            self.df.iloc[:, idxs] = self.df.iloc[:, idxs].replace(
                to_replace={
                    r"(?i)\b(yes|m)\b": 1,
                    r"(?i)\b(no|f)\b": 0
                },
                regex=True
            )

            self.df.iloc[:, idxs] = self.df.iloc[:, idxs].astype(int)
        else:
            self.df.loc[:, idxs] = self.df.loc[:, idxs].replace(
                to_replace={
                    r"(?i)\b(yes|m)\b": 1,
                    r"(?i)\b(no|f)\b": 0
                },
                regex=True
            )

            self.df.loc[:, idxs] = self.df.loc[:, idxs].astype(int)

    def column_rename(self, idxs, names=None) -> None:
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
        
        if type(idxs) is int or (type(idxs) is list and type(idxs[0]) is int):
            idxs = [self.df.columns[idx] for idx in idxs]

        replace_dic = dict([(prev, new) for prev, new in zip(idxs, names)])    
        self.df = self.df.rename(columns=replace_dic)