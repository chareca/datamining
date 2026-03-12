"""
conda activate py311ml
pip install ucimlrepo
"""
import utils

def main():
    X, y = utils.get_adults_data()
    
if __name__ == "__main__":
    main()