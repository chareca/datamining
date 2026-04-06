import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from functions_scripts.load import read_data
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class Preparador2(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pipeline = None
        pd.set_option('future.no_silent_downcasting', True)

    def __clean_text(self, X: pd.DataFrame):
        """Aplica a cada columna la minimización y borrado de espacios"""
        return X.apply(lambda x: x.str.lower().str.replace(" ", "", regex=False))

    def __categorical_to_binary(self, X: pd.DataFrame):
        """Pasa las columnas categóricas binarias a numéricas binarias con la siguiente rúbrica:
            - YES/yes -> 1
            - M/m     -> 1
            - NO/no   -> 0
            - F/f     -> 0"""
        X_copy = X.copy()
        mapping = {
            r"(?i)\b(yes|m)\b": 1,
            r"(?i)\b(no|f)\b": 0
        }
        for col in X_copy.columns:
            X_copy[col] = X_copy[col].replace(to_replace=mapping, regex=True)
            X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').astype("int8")
        return X_copy
    
    def __rename(self, X: pd.DataFrame):
        return X.rename(columns={'jundice': 'jaundice', 'austim': 'family_pdd', 'contry_of_res': 'country_of_res', 'Class/ASD': 'Class'})

    def __to_numeric(self, X: pd.DataFrame):
        return X.apply(pd.to_numeric, errors='coerce').astype(np.float32)

    def __dev_pipeline(self):
        # PASO 1: Renombrar (se pasa todo el DataFrame pero solo modifica lo indicado en __rename)
        paso_renombrar = FunctionTransformer(self.__rename)

        # PASO 2: Limpieza de texto solo en las columnas especificadas
        # Usa los nombres ya renombrados en paso_renombrar
        paso_limpiar = ColumnTransformer([
            ('clean_text', FunctionTransformer(self.__clean_text), 
             ['gender', 'ethnicity', 'jaundice', 'family_pdd', 'country_of_res', 'used_app_before', 'relation', 'Class'])
        ], remainder='passthrough', verbose_feature_names_out=False)

        # PASO 3: Binarizar solo las que son yes/no o m/f
        #         Eliminar columnas inútiles
        #         Castear numéricas
        paso_final = ColumnTransformer([
            ('binario', FunctionTransformer(self.__categorical_to_binary), ['gender', 'jaundice', 'family_pdd', 'used_app_before', 'Class']), # Binariza numéricamente columnas categoricas binarias
            ('drop', 'drop', ['id', 'age_desc']), # Borra columnas inutiles
            ('num', FunctionTransformer(self.__to_numeric), ['age']) # Castear a numérico
        ], remainder='passthrough', verbose_feature_names_out=False)

        pipe = Pipeline([
            ('renombrado', paso_renombrar),
            ('minusculas_espacios', paso_limpiar),
            ('binarizado_eliminado_casteado', paso_final)
        ])
        pipe.set_output(transform="pandas")
        return pipe

    def fit(self, X: pd.DataFrame, y=None):        
        self.pipeline = self.__dev_pipeline()

        self.pipeline.fit(X, y)
        
        return self
    
    def transform(self, X: pd.DataFrame):
        return self.pipeline.transform(X)