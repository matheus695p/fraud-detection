import warnings
# import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import (cross_val_score, RepeatedStratifiedKFold,
                                     GridSearchCV)
from src.preprocessing.preprocessing import supervised_preparation
from src.utils.visualizations import watch_classification_distribution
from src.config.config import xgboost_arguments

warnings.filterwarnings("ignore")

args = xgboost_arguments()

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
# weights = [250, 400, 550, 600, 750, 850, 1000]
# weights = [550, 590, 640, 700, 800, 1000]
weights = [640, 680, 700]

param_grid = dict(scale_pos_weight=weights, tree_method=["gpu_hist"],
                  alpha=[args.alpha, 0.01], gamma=[args.gamma, 0.5],
                  max_depth=[args.max_depth, 10, 12],
                  learning_rate=[0.01, 0.001],
                  eval_metric=["auc"])

# vamos a hacer un interación Kfold-Stratified
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=20)

# grid de búsqueda
grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    n_jobs=-1, cv=cv, scoring='recall')
grid_result = grid.fit(x_train, y_train, verbose=3)

# resultados
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
print(grid.best_params_)

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# evaluate model
scores_test = cross_val_score(model, x_test, y_test,
                              scoring='recall', cv=cv, n_jobs=-1)
print("Recall:", scores_test.mean())

# # summarize performance
# print('Mean recall:', np.mean(scores))
