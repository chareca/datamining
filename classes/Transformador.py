from sklearn.base import TransformerMixin, BaseEstimator
from typing import Literal
import numpy as np
import pandas as pd

class Transformador(TransformerMixin, BaseEstimator):
	def __init__(self, metodo_cat_num: Literal["orden"]="orden") -> None:
		opciones = ["orden"]
        
		if metodo_cat_num not in opciones:
			raise ValueError(f"[ERROR]: La opción de imputación numérica '{metodo_cat_num}' \
					no está disponible. Usar solo [{', '.join(opciones)}]")
		
		self._metodo_cat_num = metodo_cat_num

	def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
		""" No es necesario entrenar nada (al menos con el método de orden) """
		X = pd.DataFrame(X)
		y = pd.DataFrame(y)
		return self
	
	def transform(self, X: pd.DataFrame):
		Xaux = pd.DataFrame(X).copy()

		nombres_columnas_categoricas = Xaux.select_dtypes(include='category').columns
		for nombre_columna in nombres_columnas_categoricas:
			if self._metodo_cat_num == "orden":
				Xaux[nombre_columna] = Xaux[nombre_columna].cat.codes.astype("float32")
		return Xaux

if __name__ == '__main__':
	from Preparador3 import Preparador3
	from Imputer import Imputer

	df = pd.read_csv("datamining/data/Autism-Adult-Data.csv", delimiter=',')
	preparador = Preparador3()
	df = preparador.preparar(df)
	print("Sin transformacion:")
	df.info()

	imputer = Imputer()
	df = imputer.fit_transform(df)

	transformador = Transformador()
	df = transformador.fit_transform(df)

	print("Con transformacion:")
	df.info()