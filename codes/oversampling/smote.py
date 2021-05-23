import pandas as pd
from imblearn.over_sampling import SMOTE

# cargar los datos
path = "data/creditcard_featured.csv"
df = pd.read_csv(path)


# # targets y features
targets = ["class"]
features = list(df.columns)
for tar in targets:
    features.remove(tar)
x = df[features]
y = df[targets]

# técnica  de oversampling
oversample = SMOTE(sampling_strategy="auto", random_state=20)
x_oversampling, y_oversampling = oversample.fit_resample(x, y)

# concatenar y eliminar duplicados para tener toda la data junta
data_all = pd.concat([x_oversampling, y_oversampling], axis=1)
data_all = pd.concat([data_all, df], axis=0)
data_all.drop_duplicates(inplace=True)

# encontrar filas en data_all que no estan en el original, el oversampling
oversampling = data_all.merge(df, how='outer', indicator=True).loc[
    lambda x: x['_merge'] == 'left_only']
oversampling.drop(columns=['_merge'], inplace=True)


path = "data/creditcard_smote_oversampling.csv"
oversampling.to_csv(path, index=False)
