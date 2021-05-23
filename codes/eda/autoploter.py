import pandas as pd
from autoplotter import run_app

path = "data/creditcard_cleaned.csv"
df = pd.read_csv(path)

# hacer llamada al host local
run_app(df, mode="external", host="127.0.0.1", port=5000)

# esto debe retornar el puerto http://127.0.0.1:5000/ que debes copiar y pegar
# en algun explorador (ojo que esto consume CPU y RAM local)
