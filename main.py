from classes import Preparador3, Model, Imputer, Transformador
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from functions_scripts.load import read_data
from matplotlib import pyplot as plt
import pandas as pd                                                                                               

"""
    Hito 1 — Primera solución automatizada: ColumnTransformer + Pipeline + KNN
    Métrica: Accuracy      Validación: StratifiedKFold(10)

    1. Imputacion valores perdidos
    2. Transformación categóricas -> numéricas
    3. (dsp 2do hito) estandarizar
    4. KNN (en principio parámetros fijos)
"""


def get_graphs(X, y):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(10):
        col = X.columns[i]
        ct = pd.crosstab(X[col], y) # filas: agree/disagree, cols: 0/1                                   
        ct.columns = ["no", "yes"]
        ct.plot(kind="bar", ax=axes[i], legend=(i == 0))
        axes[i].set_title(f"Pregunta {i+1}")
        axes[i].set_xlabel("")
                
    fig.suptitle("Relaciones entre preguntas y clase final")                                                      
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2,2, figsize=(15, 6))
    axes = axes.ravel()
    fig.suptitle("Relaciones entre características y clase final")
    columns = ["jaundice", "family_pdd", "used_app_before", "relation"]
    for i in range(4):
        col = columns[i]
        ct = pd.crosstab(X[col], y)                                   
        ct.columns = ["no", "yes"]
        ct.plot(kind="bar", ax=axes[i], legend=(i == 0))
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")

    plt.show()

def main():
    # Configuración
    seed = 123
    test_size = 0.2

    # Leemos los datos y dejamos el dataframe más organizado
    df = Preparador3().preparar(read_data())

    # Creamos los conjuntos de datos
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].cat.codes.astype(int)   # no=0, yes=1

    get_graphs(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    modelo = Model(
        imputador=Imputer(),
        transformador=Transformador(),
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
    modelo.set_scorer("accuracy")

    modelo.fit(X_train, y_train)
    
    print(modelo.confusion_matrix(X_test, y_test))

if __name__ == "__main__":
    main()