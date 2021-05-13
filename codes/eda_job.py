import pandas as pd
from src.eda.exploratory_data_analysis import profiling_report
from src.utils.visualizations import pichart_plot
# hacer una exploración rápida con pandas profiling, dado la naturaleza de
# la prueba
path = "data/creditcard_cleaned.csv"
profiling_report(path, minimal_mode=True, dark_mode=True)

# después de analizar los .html, hay una correcta imputación de datos
# pero se notó un gran desbalance en la predicción de la clase
df = pd.read_csv(path)

# se puede ver mucho desbalance de clases en los datos, es mejor tratarlo
# como un problema de detección de anomalias
pichart_plot(df, col="class")

# pairplot de las variables
