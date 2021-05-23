import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
from src.preprocessing.preprocessing import supervised_preparation
from src.utils.visualizations import watch_classification_distribution

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

# Normalización de los datos
sc = MinMaxScaler(feature_range=(0, 1))
# training
x_train = sc.fit_transform(x_train)
# testing
x_test = sc.transform(x_test)


# ver distribución de las separaciones
watch_classification_distribution(y_train, y_test)
# [no hay sesgo en las distribuciones de los conjuntos de entrenamiento y test]

# ETAPA PARA SACAR MODELOS BASELINE DE ML [DESDE LO MÁS SIMPLE A MÁS COMPLEJO]

# regresión logistica con parametros de regularización
log_reg_params = {"penalty": ['l1', 'l2'],
                  'C': [0.001, 0.01, 0.1, 1, 10]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(x_train, y_train)
# sacar la regresión que tuvo mejor resultado y sus parámetros
log_reg = grid_log_reg.best_estimator_

# usar un k-nearest neighbors
knears_params = {"n_neighbors": list(range(2, 5, 1)), 'algorithm': [
    'auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(x_train, y_train)
# k-nearest neighbors mejor estimador
knears_neighbors = grid_knears.best_estimator_

# Support Vector Machine para clasificación
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': [
    'rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(x_train, y_train)
# svm mejor estimador
svc = grid_svc.best_estimator_

# Arbol de decisión para clasificación
tree_params = {"criterion": ["gini", "entropy"],
               "max_depth": list(range(2, 4, 1)),
               "min_samples_leaf": list(range(5, 7, 1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(x_train, y_train)
# tree mejor arbol
tree_clf = grid_tree.best_estimator_
