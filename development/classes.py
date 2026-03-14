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

class OutlierDetector(TransformerMixin, BaseEstimator):
	"""Detector y corrector de outliers por distintos métodos"""
	def __init__(self, k: float | None = 1.5, deteccion: str | None = "iqr", reemplazo: str | None = "mediana"):
		opciones_deteccion = ["iqr", "mediastd"]
		opciones_reemplazo = ["mediana", "media", "min", "max", "moda"]

		if deteccion not in opciones_deteccion:
			raise ValueError(f"[ERROR]: La opción de deteccion '{deteccion}' no está disponible. Usar solo [{', '.join(opciones_deteccion)}]")
		if reemplazo not in opciones_reemplazo:
			raise ValueError(f"[ERROR]: La opción de reemplazo '{reemplazo}' no está disponible. Usar solo [{', '.join(opciones_reemplazo)}]")
		
		self.stats = None
		self.modas = None
		self.columns = None
		self.mins = {}
		self.maxs = {}
		self.k = k
		self.deteccion = deteccion
		self.reemplazo = reemplazo

	def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
		if isinstance(X, np.ndarray):
			X = pd.DataFrame(X)
		if isinstance(y, np.ndarray):
			y = pd.DataFrame(y)

		self.stats = X.describe()
		self.columns = X.columns
				
		if self.reemplazo == "moda":
			self.modas = X.mode()
		elif self.reemplazo in ["min", "max"]: # Me guardo los minimos y maximos que NO sean outliers
			for column in self.columns:
				if self.deteccion == "iqr":
					Q1 = self.stats[column]["25%"]
					Q3 = self.stats[column]["75%"]
					IQR = Q3-Q1

					lim_sup = Q3 + self.k*IQR
					lim_inf = Q1 - self.k*IQR
				else:
					media = self.stats[column]["mean"]
					std = self.stats[column]["std"]

					lim_sup = media + self.k * std
					lim_inf = media - self.k * std

				mask_sup = X[column] > lim_sup
				mask_inf = X[column] < lim_inf

				self.mins[column] = X.loc[~(mask_sup | mask_inf), column].min()
				self.maxs[column] = X.loc[~(mask_sup | mask_inf), column].max()

		return self

	def transform(self, X: pd.DataFrame):
		if isinstance(X, np.ndarray):
			Xaux = pd.DataFrame(X).copy()
		else:
			Xaux = X.copy()

		for column in self.columns:
			if self.deteccion == "iqr":
				Q1 = self.stats[column]["25%"]
				Q3 = self.stats[column]["75%"]
				IQR = Q3-Q1

				lim_sup = Q3 + self.k*IQR
				lim_inf = Q1 - self.k*IQR
			else:
				media = self.stats[column]["mean"]
				std = self.stats[column]["std"]

				lim_sup = media + self.k * std
				lim_inf = media - self.k * std

			mask_sup = Xaux[column] > lim_sup
			mask_inf = Xaux[column] < lim_inf

			if self.reemplazo == "mediana":
				value = self.stats[column]["50%"]
			elif self.reemplazo == "media":
				value = self.stats[column]["mean"]
			elif self.reemplazo == "moda":
				value = self.modas[column][0]
			elif self.reemplazo == "min":
				value = self.mins[column]
			elif self.reemplazo == "max":
				value = self.maxs[column]

			# TODO: Decidir que hacer con las columnas int
			if Xaux.loc[:, column].dtype == int:
				value = round(value)

			Xaux.loc[mask_sup, column] = value
			Xaux.loc[mask_inf, column] = value
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
		"""Crea y devuelve un objeto de Pipeline utilizando los atributos del objeto self."""
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
		"""Funcion para guardar los parámetros de un GridSearchCV en el objeto self."""
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