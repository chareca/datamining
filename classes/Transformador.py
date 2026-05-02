from sklearn.base import TransformerMixin, BaseEstimator
from typing import Literal
import numpy as np
import pandas as pd

class Transformador(TransformerMixin, BaseEstimator):
    def __init__(self, metodo_cat_num: Literal["orden", "conteo", "ohe", "binary"]="orden") -> None:
        opciones = ["orden", "conteo", "ohe", "binary"]
        
        if metodo_cat_num not in opciones:
            raise ValueError(f"[ERROR]: La opción de imputación numérica '{metodo_cat_num}' \
                    no está disponible. Usar solo [{', '.join(opciones)}]")
        
        self.metodo_cat_num = metodo_cat_num

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
        """ No es necesario entrenar nada (al menos con el método de orden) """
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        return self
    
    def transform(self, X: pd.DataFrame):
        Xaux = pd.DataFrame(X).copy()
        
        nombres_columnas_categoricas = Xaux.select_dtypes(include='category').columns
        for nombre_columna in nombres_columnas_categoricas:
            if self.metodo_cat_num == "orden":
                Xaux[nombre_columna] = Xaux[nombre_columna].cat.codes.astype("float32")
            elif self.metodo_cat_num == "conteo":
                contador = Xaux[nombre_columna].value_counts()
                Xaux[nombre_columna] = Xaux[nombre_columna].map(contador).astype("float32")
            elif self.metodo_cat_num == "ohe":
                dummies = pd.get_dummies(Xaux[nombre_columna], prefix=nombre_columna, dtype="float32")
                Xaux = pd.concat([Xaux.drop(columns=[nombre_columna]), dummies], axis=1)
            elif self.metodo_cat_num == "binary":
                codes = Xaux[nombre_columna].cat.codes.to_numpy()
                max_bits = int(np.ceil(np.log2(codes.max() + 1)))
                binary_matrix = ((codes[:, None] & (1 << np.arange(max_bits))) > 0).astype(np.float32)
                binary_df = pd.DataFrame(
                    binary_matrix,
                    columns=[f"{nombre_columna}_bit_{i}" for i in range(max_bits)],
                    index=Xaux.index
                )
                Xaux = Xaux.drop(columns=[nombre_columna])
                Xaux = pd.concat([Xaux, binary_df], axis=1)
        return Xaux

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

    transformador = Transformador(metodo_cat_num="orden")
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