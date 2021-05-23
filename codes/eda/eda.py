import pandas as pd
import seaborn as sns
from src.eda.exploratory_data_analysis import profiling_report
from src.utils.visualizations import (piechart_plot, pairplot_df,
                                      correlation_matrix)
# hacer una exploración rápida con pandas profiling, dado la naturaleza de
# la prueba
path = "data/creditcard_cleaned.csv"
profiling_report(path, minimal_mode=True, dark_mode=True)

# después de analizar los .html, hay una correcta imputación de datos
# pero se notó un gran desbalance en la predicción de la clase
df = pd.read_csv(path)

# se puede ver mucho desbalance de clases en los datos, es mejor tratarlo
# como un problema de detección de anomalias
piechart_plot(df, col="class")
sns.countplot(x='class', data=df)


# ver correlaciones
correlation_matrix(df, method="pearson")
correlation_matrix(df, method="spearman")
correlation_matrix(df, method="kendall")

# [NO ES NECESARIO DE ACUERDO A LAS MATRICES DE CORRELACIONES]
# pairplot de las variables
for col1 in df.columns:
    for col2 in df.columns:
        if col1 == col2:
            pass
        else:
            print("Scatter plot para ver correlaciones")
            pairplot_df(df, col1, col2, path="results/scatter-plot")
