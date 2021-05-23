import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (recall_score, roc_curve, confusion_matrix,
                             classification_report)
from src.preprocessing.preprocessing import supervised_preparation
from src.utils.visualizations import (watch_classification_distribution,
                                      training_history, plot_confusion_matrix)
from src.config.config import nn_arguments

# cargar argumentos de la red
args = nn_arguments()
# cargar datos
path = "data/creditcard_featured.csv"
df = pd.read_csv(path)
# cargar datos oversampling
path = "data/creditcard_oversampling.csv"
oversampling = pd.read_csv(path)

# # targets y features
targets = ["class"]
features = list(df.columns)
for tar in targets:
    features.remove(tar)
x = df[features]
y = df[targets]

# data oversampling
x_over = oversampling[features]
y_over = oversampling[targets]

# división del conjutno de datos
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=args.validation_size, stratify=y,
    random_state=args.random_state)

# trabajar con numpy array
x_train, y_train = supervised_preparation(x_train, y_train)
x_test, y_test = supervised_preparation(x_test, y_test)

# concatenar la data oversampling solo en el conjunto de entrenamiento
# esta no puede ser usada en el conjunto de test dado que es data ficticia
x_train = np.concatenate([x_train, x_over], axis=0)
y_train = np.concatenate([y_train, y_over], axis=0)

# concatenar para randomizar los datos
x = np.concatenate([x_train, y_train], axis=1)
x = pd.DataFrame(x)
x = x.sample(frac=1).to_numpy()
# separar entre x_train e y_train
x_train = x[:, 0:-1]
y_train = x[:, -1]
y_train = np.reshape(y_train, (-1, 1))

# Normalización de los datos
sc = MinMaxScaler(feature_range=(0, 1))
# training
x_train = sc.fit_transform(x_train)
# testing
x_test = sc.transform(x_test)

# ver distribución de las separaciones
watch_classification_distribution(y_train, y_test)

# crear arquitectura del modelo
nn = tf.keras.Sequential()
nn.add(tf.keras.layers.Dense(1024, input_dim=x_train.shape[1]))
nn.add(tf.keras.layers.BatchNormalization())
nn.add(tf.keras.layers.Activation("relu"))
nn.add(tf.keras.layers.Dropout(0.3))
nn.add(tf.keras.layers.Dense(128))
nn.add(tf.keras.layers.BatchNormalization())
nn.add(tf.keras.layers.Activation("relu"))
nn.add(tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid'))

# arquitectura usada
nn.summary()
nn.compile(loss=args.loss_function, optimizer=args.optimizer)

# penalization de función de costos
weights = {0: 1, 1: 1}
# llamar callbacks de early stopping
tf.keras.callbacks.Callback()

# condición de parada
stop_condition = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', patience=args.patience, verbose=1,
    min_delta=args.min_delta, restore_best_weights=True)

# bajar el learning_rate durante la optimización
learning_rate_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=args.lr_factor,
    patience=args.lr_patience,
    verbose=1,
    mode="auto",
    cooldown=0,
    min_lr=args.lr_min)

# cuales son los callbacks que se usaran
callbacks = [stop_condition, learning_rate_schedule]

# entrenar la red
history = nn.fit(x_train, y_train,
                 validation_split=0.2,
                 batch_size=args.batch_size,
                 epochs=args.epochs,
                 shuffle=False,
                 verbose=1,
                 callbacks=callbacks,
                 class_weight=weights)

# ver resutados de entrenamiento
training_history(history, model_name="NN", filename="NN")

# evaluar
y_pred = nn.predict(x_train)
# thershold
fpr, tpr, thresholds = roc_curve(y_train, y_pred)
# calcular g-mean para cada thershold
gmeans = np.sqrt(tpr * (1-fpr))
# localizar indices de los máximos gmen
ix = np.argmax(gmeans)
print(f'Mejor theshold={thresholds[ix]}, G-Mean={gmeans[ix]}')

# buscar el mejor thesh
thresh = thresholds[ix]
thresh = 0.5
y_pred = nn.predict(x_test)
y_pred = pd.DataFrame(y_pred, columns=["pred"])
y_pred["pred"] = y_pred["pred"].apply(lambda x: 1 if x > thresh else 0)
y_pred = y_pred.to_numpy()
recall = recall_score(y_test, y_pred.round(), average="binary")
print(thresh, recall)

class2idx = {"legitima": 0, "fraude": 1}
idx2class = {v: k for k, v in class2idx.items()}

confusion_matrix_df = pd.DataFrame(confusion_matrix(
    y_test, y_pred)).rename(columns=idx2class, index=idx2class)

plot_confusion_matrix(confusion_matrix_df, cmap=plt.cm.hot)

# reporte de clasificación
class_report = classification_report(y_test, y_pred)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred.round())
(tn, fp, fn, tp) = cm.flatten()
print("Recall:", tp / (tp + fn) * 100)
print("Precision:", tp / (tp + fp) * 100)
