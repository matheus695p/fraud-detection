import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import (cross_val_score, RepeatedStratifiedKFold,
                                     GridSearchCV)
from src.preprocessing.preprocessing import supervised_preparation
from src.utils.visualizations import watch_classification_distribution

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

# división del conjutno de datos
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2,
    stratify=y, random_state=20)

# trabajar con numpy array
x_train, y_train = supervised_preparation(x_train, y_train)
x_test, y_test = supervised_preparation(x_test, y_test)

# normalización de los datos
sc = MinMaxScaler(feature_range=(0, 1))
# training
x_train = sc.fit_transform(x_train)
# testing
x_test = sc.transform(x_test)

# ver distribución de las separaciones
watch_classification_distribution(y_train, y_test)
# [no hay sesgo en las distribuciones de los conjuntos de entrenamiento y test]

# modelo como clasificador
model = XGBClassifier()

# crear el clasificador con xgboost [0.17 % la clase desbalanceada]
# ==> scale_pos_weight = 588 = 100/0.17, pero se va a buscar a prueba y error
weights = [250, 400, 550, 600, 750, 850, 1000]
param_grid = dict(scale_pos_weight=weights)

# vamos a hacer un interación Kfold-Stratified
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# grid de búsqueda
grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    n_jobs=-1, cv=cv, scoring='roc_auc')
grid_result = grid.fit(x_train, y_train)

# resultados
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# evaluate model
# scores = cross_val_score(model, x_train, y_train,
#                          scoring='recall', cv=cv, n_jobs=-1)
# # summarize performance
# print('Mean recall:', np.mean(scores))
