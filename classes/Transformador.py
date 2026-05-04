import numpy as np
import pandas as pd

from typing import Literal
from sklearn.base import TransformerMixin, BaseEstimator

class Transformador(TransformerMixin, BaseEstimator):

    def __init__(self, metodo_cat_num: Literal["orden", "conteo", "ohe", "binary"]="orden") -> None:
        opciones = ["orden", "conteo", "ohe", "binary"]
        
        if metodo_cat_num not in opciones:
            raise ValueError(f"[ERROR]: La opción de imputación numérica '{metodo_cat_num}' \
                    no está disponible. Usar solo [{', '.join(opciones)}]")
        
        self.metodo_cat_num = metodo_cat_num
        self.correspondencias = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        columnas_cat = X.select_dtypes(include='category').columns
        for col in columnas_cat:
            if self.metodo_cat_num == "orden":
                mapping = {}
                categorias = X[col].cat.categories
                for i, cat in enumerate(categorias):
                    mapping[cat] = i
                self.correspondencias[col] = mapping

            elif self.metodo_cat_num == "conteo":
                mapping = {}
                conteo = X[col].value_counts()
                for cat in conteo.index:
                    mapping[cat] = conteo[cat]
                self.correspondencias[col] = mapping

            elif self.metodo_cat_num == "ohe":
                categorias = list(X[col].cat.categories)
                self.correspondencias[col] = categorias

            elif self.metodo_cat_num == "binary":
                mapping = {}
                categorias = list(X[col].cat.categories)
                for i, cat in enumerate(categorias):
                    mapping[cat] = i

                # Bits necesarios
                n = len(categorias)
                bits = int(np.ceil(np.log2(n))) if n > 1 else 1
                self.correspondencias[col] = {
                    "mapping": mapping,
                    "bits": bits
                }

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        # Categorical columns cannot accept new values via assignment; convert to object first
        for col in X.select_dtypes("category").columns:
            X[col] = X[col].astype(object)

        cols_to_drop = []
        new_cols = {}

        for col in list(self.correspondencias.keys()):
            if self.metodo_cat_num in ["orden", "conteo"]:
                mapping = self.correspondencias[col]
                X[col] = X[col].map(mapping)

            elif self.metodo_cat_num == "ohe":
                categorias = self.correspondencias[col]
                for cat in categorias:
                    new_cols[f"{col}_{cat}"] = (X[col] == cat).astype(int)
                cols_to_drop.append(col)

            elif self.metodo_cat_num == "binary":
                info = self.correspondencias[col]
                mapping = info["mapping"]
                bits = info["bits"]

                numeros = X[col].map(mapping).fillna(0).astype(int).to_numpy()
                for i in range(bits):
                    new_cols[f"{col}_bit_{i}"] = (numeros >> i) & 1
                cols_to_drop.append(col)

        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        if new_cols:
            X = pd.concat([X, pd.DataFrame(new_cols, index=X.index)], axis=1)

        return X
    
if __name__ == '__main__':
    from Preparador import Preparador
    from Imputer import Imputer

    df = pd.read_csv("datamining/data/Autism-Adult-Data.csv", delimiter=',')
    preparador = Preparador()
    df = preparador.preparar(df)
    print("=============================")
    print("    SIN TRANSFORMACION")
    print("=============================")

    imputer = Imputer()
    df = imputer.fit_transform(df)
    
    print("------------------ INFO DATASET ------------------")
    df.info()

    print("------------------ VALORES DE COLUMNAS ------------------")
    for col in df.columns:
        print(col, df[col].unique())

    transformador = Transformador(metodo_cat_num="binary")
    df = transformador.fit_transform(df)

    print("\n=============================")
    print("    CON TRANSFORMACION")
    print("=============================")
    
    
    print("------------------ INFO DATASET ------------------")
    df.info()
    
    print("------------------ VALORES DE COLUMNAS ------------------")
    for col in df.columns:
        print(col, df[col].unique())
    print(df.head())