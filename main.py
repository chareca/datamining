"""
1. Imputacion valores perdidos
2. Transformación categóricas -> numéricas
3. (dsp 2do hito) estandarizar
4. KNN (en principio parámetros fijos)
"""
from classes import OutlierDetector, Preparador, Preparador2, Preparador3, Model, Imputer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from functions_scripts.load import read_data
from sklearn.metrics import accuracy_score


# Juan y Eloy les recomiendo las extensiones de VSCode 
# "TODO Highlight" de "Wayou Liu"
    # te resalta todos los TODOs en el proyecto
# Todo Tree
    # Te agrupa todos los TODOs en el proyecto en una tab a la izquierda

def main():
    seed = 123
    test_size = 0.2

    df = read_data()
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    modelo = Model(
        preparador=Preparador(),
        imputador=Imputer(),
        # Falta transformador
        # transformador=Transformer(),
        clasificador=KNeighborsClassifier(),
        estandarizador=StandardScaler()
    )


    param_grid = {
        "imputador__metodo_imputacion_vars_num": ["media", "mediana"],
        "imputador__metodo_imputacion_vars_cat": ["moda", "missing"],
        "modelo__n_neighbors": [1, 2, 3, 5, 8],
        "modelo__weights": ["uniform", "distance"],
        "modelo__p": [1,2],
        "modelo__n_jobs": [-1],
    }
    modelo.set_params(param_grid)
    modelo.set_scorer(accuracy_score)

    modelo.fit(X_train, y_train)    
    
    print(modelo.confusion_matrix(X_test, y_test))

if __name__ == "__main__":
    main()