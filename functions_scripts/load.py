import pandas as pd

def read_data():
    df = pd.read_csv('data/Autism-Adult-Data.csv', delimiter=',')
    return df