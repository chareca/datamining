"""
1. Imputacion valores perdidos
2. Transformación categóricas -> numéricas
3. (dsp 2do hito) estandarizar
4. KNN (en principio parámetros fijos)
"""
import utils
from classes import OutlierDetector
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Juan y Eloy les recomiendo las extensiones de VSCode 
# "TODO Highlight" de "Wayou Liu"
    # te resalta todos los TODOs en el proyecto
# Todo Tree
    # Te agrupa todos los TODOs en el proyecto en una tab a la izquierda

# TODO: Eliminar columna fnlwgt con ColumnTransformer

def main():
    X, y = utils.get_adults_data()
    
    print(X.head())

if __name__ == "__main__":
    main()