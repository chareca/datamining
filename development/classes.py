"""Clases utilizadas en los demás archivos"""
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer

class OutlierDetector(TransformerMixin, BaseEstimator): # TODO: Generalizar a más metodos
	"""Detector y corrector de outliers por la mediana"""
	def __init__(self, k: float | None = 1.5):
		self.stats = None
		self.columns = None
		self.k = k

	def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
		if isinstance(X, np.ndarray):
			X = pd.DataFrame(X)
		if isinstance(y, np.ndarray):
			y = pd.DataFrame(y)
		
		self.stats = X.describe()
		self.columns = X.columns
		return self

	def transform(self, X: pd.DataFrame):
		if isinstance(X, np.ndarray):
			Xaux = pd.DataFrame(X).copy()
		else:
			Xaux = X.copy()

		for column in self.columns:
			Q1 = self.stats[column]["25%"]
			Q3 = self.stats[column]["75%"]
			IQR = Q3-Q1

			lim_sup = Q3 + self.k*IQR
			lim_inf = Q1 - self.k*IQR

			mask_sup = Xaux[column] > lim_sup
			mask_inf = Xaux[column] < lim_inf

			Xaux.loc[mask_sup, column] = self.stats[column]["50%"]
			Xaux.loc[mask_inf, column] = self.stats[column]["50%"]

		return Xaux
  
class Model(): # TODO: Agregar ColumnTransformer
	def __init__(self, imputador, transformador, clasificador, estandarizador = None):
		self.imputador = imputador
		self.transformador = transformador
		self.estandarizador = estandarizador
		self.clasificador = clasificador
		self.param_grid = None
		self.scorer = None
		self.modelo = None

	def create_pipeline(self):
		if self.estandarizador is not None:
			pipe = Pipeline([("imputador", self.imputador),
					("transformador", self.transformador),
					("estandarizador", self.estandarizador),
					("modelo", self.clasificador)])
		else:
			pipe = Pipeline([("imputador", self.imputador),
					("transformador", self.transformador),
					("modelo", self.clasificador)])
		return pipe
		
	def set_params(self, param_grid: dict) -> None:
		# Verificación datos válidos del param_grid
		if isinstance(param_grid, list):
			for params in param_grid:
				for key in params.keys():
					if key.split("__")[0] not in ["imputador", "transformador", "estandarizador", "modelo"]:
						raise ValueError("[ERROR]: El diccionario de parametros introducido utiliza nombres inválidos.\nUsar: 'imputador', 'transformador', 'estandarizador' o 'modelo'.")
		else:
			for key in param_grid.keys():
				if key.split("__")[0] not in ["imputador", "transformador", "estandarizador", "modelo"]:
					raise ValueError("[ERROR]: El diccionario de parametros introducido utiliza nombres inválidos.\nUsar: 'imputador', 'transformador', 'estandarizador' o 'modelo'.")
		self.param_grid = param_grid

	def set_scorer(self, scorer):
		self.scorer = scorer

	def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | None = None):
		"""Devuelve el modelo entrenado."""
		
		pipe = self.create_pipeline()
		self.modelo = pipe

		if self.param_grid is not None:
			grid = GridSearchCV(pipe, self.param_grid, scoring=self.scorer)
			self.modelo = grid
		
		self.modelo = self.modelo.fit(X, y)
		return self.modelo
		
	def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
		return self.modelo.predict(X)

	def accuracy_score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, decimals: int | None = 4) -> float:
		predicciones = self.predict(X)
		return round(accuracy_score(y, predicciones)*100, decimals)
	
	def f1_score(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, decimals: int | None = 4) -> float:
		predicciones = self.predict(X)
		return round(f1_score(y, predicciones)*100, decimals)
		
	def confusion_matrix(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame, labels: list | None = None) -> np.ndarray:
		predicciones = self.predict(X)
		return confusion_matrix(y, predicciones, labels)