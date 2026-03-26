from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd

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