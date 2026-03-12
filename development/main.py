"""
conda activate py311ml
pip install ucimlrepo
"""
from ucimlrepo import fetch_ucirepo 
from utils import *

def main():
  
    # fetch dataset 
    adult = fetch_ucirepo(id=2) 
    
    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets 
    
    # metadata 
    print(adult.metadata) 
    
    # variable information 
    print(adult.variables) 


    print(X.head())
    print(y.head())

if __name__ == "__main__":
    main()