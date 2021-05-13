import pandas as pd
from src.preprocessing.cleaning import (renaming_columns, convert_df_float,
                                        drop_spaces_data, replace_empty_nans)

# lectura de los datos
path = "data/creditcard.csv"
df = pd.read_csv(path)

# correr stack de limpieza previa para asegurar buena formatación de los datos
# en el caso de que hayan vacios en los comienzos de columnas categoricas
df = drop_spaces_data(df)
# reemplaza vacios con nans, para identificar vacios en el dataframe
df = replace_empty_nans(df)

# En el caso de que la salida de este print sea cero nans pasar a las otras
# funciones
print(df.isna().sum())

# convierte las columans que se pueden a float
df = convert_df_float(df)
# esta función es un trastorno obsesivo compulsivo de no tener columnas con
# espacios o con mayusculas (por PEP8 y por los problemas que trae en bases
# de datos)
df = renaming_columns(df)
# mirada estadistica
describe = df.describe()

head = df.head(100)

path_save = "data/creditcard_cleaned.csv"
df.to_csv(path_save, index=False)
