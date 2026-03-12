""" Funciones principales y auxiliares """
from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd


def get_adults_data(return_full = False):
    adult = fetch_ucirepo(id=2) 
    
    X = adult.data.features 
    y = adult.data.targets 

    if return_full:
        return X, y, adult
    else:
        return X, y