import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.base import TransformerMixin, BaseEstimator

class Model(TransformerMixin, BaseEstimator): # TODO: Agregar ColumnTransformer
	def __init__(self, preparador, imputador, transformador, clasificador, estandarizador = None):
		self.preparador = preparador
		self.imputador = imputador
		self.transformador = transformador
		self.estandarizador = estandarizador
		self.clasificador = clasificador
		self.param_grid = None
		self.scorer = None
		self.modelo = None

	def create_pipeline(self):
		"""Crea y devuelve un objeto de Pipeline utilizando los atributos del objeto self"""
		if self.estandarizador is not None:
			pipe = Pipeline([
				    ("preparador", self.preparador),
                    ("imputador", self.imputador),
					("transformador", self.transformador),
					("estandarizador", self.estandarizador),
					("modelo", self.clasificador)])
		else:
			pipe = Pipeline([
				    ("preparador", self.preparador),
				    ("imputador", self.imputador),
					("transformador", self.transformador),
					("modelo", self.clasificador)])
		return pipe
		
	def set_params(self, param_grid: dict) -> None:
		"""Funcion para guardar los parámetros de un GridSearchCV en el objeto self."""
		# Verificación datos válidos del param_grid
		if isinstance(param_grid, list):
			for params in param_grid:
				for key in params.keys():
					if key.split("__")[0] not in ["preparador", "imputador", "transformador", "estandarizador", "modelo"]:
						raise ValueError("[ERROR]: El diccionario de parametros introducido utiliza nombres inválidos.\nUsar: 'preparador', 'imputador', 'transformador', 'estandarizador' o 'modelo'.")
		else:
			for key in param_grid.keys():
				if key.split("__")[0] not in ["preparador", "imputador", "transformador", "estandarizador", "modelo"]:
					raise ValueError("[ERROR]: El diccionario de parametros introducido utiliza nombres inválidos.\nUsar: 'preparador', 'imputador', 'transformador', 'estandarizador' o 'modelo'.")
		self.param_grid = param_grid

	def set_scorer(self, scorer):
		"""Modifica el scorer utilizado por el modelo."""
		self.scorer = scorer

	def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | None = None):
		"""Entrena, guarda y devuelve el modelo entrenado.
		Utiliza pipeline, y en caso de haber parámetros en self.param_grid, utiliza un GridSearchCV con el Pipeline."""
		
		pipe = self.create_pipeline()
		self.modelo = pipe

		if self.param_grid is not None:
			grid = GridSearchCV(pipe, self.param_grid, scoring=self.scorer)
			self.modelo = grid
		
		self.modelo = self.modelo.fit(X, y)
		return self.modelo
		
	def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
		"""Predice X con el modelo del objeto."""
		return self.modelo.predict(X)

	def accuracy_score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, decimals: int | None = 4) -> float:
		"""Calcula el accuracy_score para las predicciones de los datos de X con y como y_true."""
		predicciones = self.predict(X)
		return round(accuracy_score(y, predicciones)*100, decimals)
	
	def f1_score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, decimals: int | None = 4) -> float:
		"""Calcula el f1-score para las predicciones de los datos de X con y como y_true."""
		predicciones = self.predict(X)
		return round(f1_score(y, predicciones)*100, decimals)
		
	def confusion_matrix(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, labels: list | None = None) -> np.ndarray:
		"""Calcula la matriz de confusión para las predicciones de los datos de X con y como y_true."""
		predicciones = self.predict(X)
		return confusion_matrix(y, predicciones, labels)