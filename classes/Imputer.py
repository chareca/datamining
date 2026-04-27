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
		
		self.metodo_imputacion_vars_num = metodo_imputacion_vars_num
		self.metodo_imputacion_vars_cat = metodo_imputacion_vars_cat

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
				if np.where(Xaux[nombre_columna].isna())[0].size > 0:
					nombres_columnas_objetivo.append(nombre_columna)
			else:
				if Xaux[nombre_columna].isna().any():
					nombres_columnas_objetivo.append(nombre_columna)
		
		# Imputar valores en las columnas objetivo
		for nombre_columna in nombres_columnas_objetivo:
			if Xaux[nombre_columna].dtype == "category":
				Xaux[nombre_columna] = Xaux[nombre_columna].astype("object")
				if self.metodo_imputacion_vars_cat == "moda":
					moda_columna = Xaux.loc[~Xaux[nombre_columna].isna(), nombre_columna].mode()[0]
					Xaux.loc[Xaux[nombre_columna].isna(), nombre_columna] = moda_columna
				elif self.metodo_imputacion_vars_cat == "missing":
					Xaux.loc[Xaux[nombre_columna].isna(), nombre_columna] = "missed"
				Xaux[nombre_columna] = Xaux[nombre_columna].astype("category")
			else:
				if self.metodo_imputacion_vars_num == "media":
					media_columna = np.nanmean(Xaux[nombre_columna])
					Xaux.loc[Xaux[nombre_columna].isna(), nombre_columna] = media_columna
				elif self.metodo_imputacion_vars_num == "mediana":
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