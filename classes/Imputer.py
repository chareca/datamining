from sklearn.base import TransformerMixin, BaseEstimator
from typing import Literal
import numpy as np
import pandas as pd

class Imputer(TransformerMixin, BaseEstimator):
	"""Imputador de valores perdidos para variables categóricas y numéricas por distintos métodos"""
	def __init__(
			self,
			metodo_imputacion_vars_num: Literal["media", "mediana"]="mediana",
			metodo_imputacion_vars_cat: Literal["moda", "missing"]="missing"
		) -> None:

		opciones_met_imp_vars_num = ["media", "mediana"]
		opciones_met_imp_vars_cat = ["moda", "missing"]
        
		if metodo_imputacion_vars_num not in opciones_met_imp_vars_num:
			raise ValueError(f"[ERROR]: La opción de imputación numérica '{metodo_imputacion_vars_num}' \
					no está disponible. Usar solo [{', '.join(opciones_met_imp_vars_num)}]")
		if metodo_imputacion_vars_cat not in opciones_met_imp_vars_cat:
			raise ValueError(f"[ERROR]: La opción de imputación categórica '{metodo_imputacion_vars_cat}' \
					no está disponible. Usar solo [{', '.join(opciones_met_imp_vars_cat)}]")
		
		self._metodo_numericas = metodo_imputacion_vars_num
		self._metodo_categoricas = metodo_imputacion_vars_cat

	def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
		""" No es necesario entrenar nada, solo comprobar que son DF o algo equivalente"""
		X = pd.DataFrame(X)
		y = pd.DataFrame(y)
		return self
	
	def transform(self, X: pd.DataFrame):
		Xaux = pd.DataFrame(X).copy()

		# Localizar las columnas en las que hay que imputar
		nombres_columnas_objetivo = []
		for nombre_columna in Xaux.columns:
			if Xaux[nombre_columna].dtype == "category":
				if np.where(Xaux[nombre_columna] == "?")[0].size > 0:
					nombres_columnas_objetivo.append(nombre_columna)
			else:
				if Xaux[nombre_columna].isna().any():
					nombres_columnas_objetivo.append(nombre_columna)
		
		# Imputar valores en las columnas objetivo
		for nombre_columna in nombres_columnas_objetivo:
			if Xaux[nombre_columna].dtype == "category":
				if self._metodo_categoricas == "moda":
					moda_columna = Xaux[Xaux[nombre_columna] != "?", nombre_columna].mode()[0]
					Xaux.loc[Xaux[nombre_columna] == "?", nombre_columna] = moda_columna
				elif self._metodo_categoricas == "missing":
					pass # No hacemos nada, las que son valores perdidos "?" simplemente las tratamos como tal
			else:
				if self._metodo_numericas == "media":
					media_columna = np.nanmean(Xaux[nombre_columna])
					Xaux.loc[Xaux[nombre_columna].isna(), nombre_columna] = media_columna
				elif self._metodo_numericas == "mediana":
					mediana_columna = np.nanmedian(Xaux[nombre_columna])
					Xaux.loc[Xaux[nombre_columna].isna(), nombre_columna] = mediana_columna

		return Xaux

if __name__ == '__main__':
	from Preparador3 import Preparador3

	df = pd.read_csv("datamining/data/Autism-Adult-Data.csv", delimiter=',')
	preparador = Preparador3()
	df = preparador.preparar(df)

	print("Sin imputación:")
	print("age: ", np.unique(df["age"]))
	print("relation: ", np.unique(df["relation"]))
	print("ethnicity: ", np.unique(df["ethnicity"]))

	imputer = Imputer()
	df = imputer.fit_transform(df)

	print("Con imputación:")
	print("age: ", np.unique(df["age"]))
	print("relation: ", np.unique(df["relation"]))
	print("ethnicity: ", np.unique(df["ethnicity"]))