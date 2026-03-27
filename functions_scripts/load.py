import pandas as pd

def read_data(filepath):
    try:
        df = pd.read_csv(filepath, delimiter=',')
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No se encontro el archivo: {filepath}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"El archivo esta vacio: {filepath}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Formato CSV invalido en: {filepath}") from exc

    df = df.rename(columns={"jundice": "jaundice"})
    return df
