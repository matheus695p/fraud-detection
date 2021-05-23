import numpy as np
import pandas as pd
from src.analytics.transformations import (exponential_features,
                                           polinomial_features,
                                           log_features,
                                           selection_by_correlation)

# leer datos
path = "data/creditcard_cleaned.csv"
df = pd.read_csv(path)

# columnas a hacer feature engeniering
columns = list(df.columns)

# columnas de predicción
targets = ["class"]
for col in targets:
    columns.remove(col)

# exponencial
df = exponential_features(df, columns)
# logaritmo
df = log_features(df, columns)
# raiz
df = polinomial_features(df, columns, grade=0.5)
# potencia 2
df = polinomial_features(df, columns, grade=2)
# potencia 3
df = polinomial_features(df, columns, grade=3)
# potencia 5
df = polinomial_features(df, columns, grade=5)

# seleccionar por correlación y con método lineal para eliminar colinealidad
df = selection_by_correlation(df, threshold=0.5, method="pearson")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# droping columns
nans = pd.DataFrame(df.isna().sum(), columns=["contador"])
nans.reset_index(drop=False, inplace=True)
nans.rename(columns={"index": "column"}, inplace=True)
nans["porcentaje"] = nans["contador"] / len(df) * 100

# no dejar columnas con nans
droping_cols = nans[nans["porcentaje"] > 0]["column"].to_list()
df.drop(columns=droping_cols, inplace=True)

# guardar dfframe
path_output = "data/creditcard_featured.csv"
df.to_csv(path_output, index=False)
