""" Clases utilizadas en los demás archivos """
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class OutlierDetector(TransformerMixin, BaseEstimator):
	""" Detector y corrector de outliers por la mediana """
	def __init__(self, k: float | None = 1.5):
		self.stats = None
		self.columns = None
		self.k = k

	def fit(self, X: pd.DataFrame, y: pd.DataFrame | None = None):
		self.stats = X.describe()
		self.columns = X.columns
		return self

	def transform(self, X: pd.DataFrame):
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
  
class Model():
	def __init__(self, clasificador, imputador, transformador, estandarizador = None):
		self.clasificador = clasificador
		self.estandarizador = estandarizador
		self.imputador = imputador
		self.transformador = transformador

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
	