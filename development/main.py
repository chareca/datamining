"""
conda activate py311ml
pip install ucimlrepo
"""

"""
1. Imputacion valores perdidos
2. Transformación categóricas -> numéricas
3. (dsp 2do hito) estandarizar
4. KNN (en principio parámetros fijos)
"""
import utils
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def main():
    X, y = utils.get_adults_data()

if __name__ == "__main__":
    main()