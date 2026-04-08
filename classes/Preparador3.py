import numpy as np
import pandas as pd

class Preparador3():
    def __init__(self):
        pass

    def preparar(self, X: pd.DataFrame):
        X = self.__rename(X)
        X = self.__column_del(X)
        X = self.__invert_meaning_answers(X)
        X = self.__clean_text(X)
        X = self.__type_variables(X)
        return X
    
    def __rename(self, X: pd.DataFrame):
        """Pasa todas las columnas a minuscula y renombra las columnas con nombres erróneos"""
        Xaux = X.rename(columns=dict((col, col.lower()) for col in X.columns))
        Xaux = Xaux.rename(columns={'jundice': 'jaundice', 'austim': 'family_pdd', 'contry_of_res': 'country_of_res', 'class/asd': 'class'})
        return Xaux
    
    def __column_del(self, X: pd.DataFrame):
        """ Elimina las columnas """
        X = X.drop(columns=["id", "result", "age_desc"])
        return X

    def __invert_meaning_answers(self, X: pd.DataFrame):
        """ En info/Dataset_Autism.pdf se explica que las siguientes columnas tienen los valores "al revés" """
        cols = ["a2_score", "a3_score", "a4_score", "a5_score", "a6_score", "a9_score"]
        X[cols] = np.abs(X[cols] - 1)
        return X
    
    def __type_variables(self, X: pd.DataFrame):
        """ Pasa las variables a categoricas y numéricas a dichos tipos """
        cols_score = ["a1_score", "a2_score", "a3_score", "a4_score", "a5_score",
                "a6_score", "a7_score", "a8_score", "a9_score", "a10_score"]
        X[cols_score] = np.where(X[cols_score] == 1, "agree", "disagree")

        cols_cat = ["a1_score", "a2_score", "a3_score", "a4_score", "a5_score",
                "a6_score", "a7_score", "a8_score", "a9_score", "a10_score",
                "gender", "ethnicity", "jaundice", "family_pdd", "country_of_res",
                "used_app_before", "relation", "class"]
        X[cols_cat] = X[cols_cat].astype('category')

        cols_num = ["age"]
        X[cols_num] = X[cols_num].apply(pd.to_numeric, errors='coerce').astype(np.float32)
        return X

    def __clean_text(self, X: pd.DataFrame):
        """Aplica a cada columna la minimización y borrado de espacios"""
        cols = ['gender', 'ethnicity', 'jaundice', 'family_pdd', 'country_of_res', 'used_app_before', 'relation', 'class']
        X[cols] = X[cols].apply(lambda x: x.str.lower().str.replace(" ", "", regex=False))
        return X

if __name__ == '__main__':
    df = pd.read_csv("datamining/data/Autism-Adult-Data.csv", delimiter=',')
    preparador = Preparador3()

    print("Antes:")
    df.info()
    
    df = preparador.preparar(df)

    print("\nDespués:")
    df.info()

    print("\nVariables únicas:")
    print("a1_score...a10_score: ", np.unique(df["a1_score"]))
    print("age: ", np.unique(df["age"]))
    print("gender: ", np.unique(df["gender"]))
    print("ethnicity: ", np.unique(df["ethnicity"]))
    print("jaundice: ", np.unique(df["jaundice"]))
    print("family_pdd: ", np.unique(df["family_pdd"]))
    print("country_of_res: ", np.unique(df["country_of_res"]))
    print("used_app_before: ", np.unique(df["used_app_before"]))
    print("relation: ", np.unique(df["relation"]))
    print("class: ", np.unique(df["class"]))