import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (recall_score, confusion_matrix)
from src.preprocessing.preprocessing import supervised_preparation
from src.config.config import autoencoder_arguments
from src.utils.visualizations import (watch_classification_distribution,
                                      training_history,
                                      latent_space_visualization)
from src.analytics.metrics import mad_score
# cargar argumentos de la red
args = autoencoder_arguments()
# cargar datos
path = "data/creditcard_featured.csv"
df = pd.read_csv(path)


# # targets y features
targets = ["class"]
features = list(df.columns)
for tar in targets:
    features.remove(tar)
x = df[features]
y = df[targets]


# división del conjunto de datos
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=args.validation_size, stratify=y,
    random_state=args.random_state)

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

# Dimensión de entrada del autoencoder
input_layer = tf.keras.layers.Input(shape=(x_train.shape[1], ))

# Encoder
encoder = tf.keras.layers.Dense(48, activation="relu")(input_layer)
encoder = tf.keras.layers.Dropout(0.2)(encoder)
encoder = tf.keras.layers.Dense(32, activation='relu')(encoder)
encoder = tf.keras.layers.Dense(16, activation="relu")(encoder)
# Latent space
z = tf.keras.layers.Dense(4, activation="relu")(encoder)
# Decoder
decoder = tf.keras.layers.Dense(16, activation='relu')(z)
decoder = tf.keras.layers.Dense(32, activation='relu')(decoder)
decoder = tf.keras.layers.Dropout(0.2)(decoder)
decoder = tf.keras.layers.Dense(48, activation='relu')(decoder)
decoder = tf.keras.layers.Dense(64, activation='relu')(decoder)

# Autoencoder
autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)
# resumen
autoencoder.summary()

# grafo estatico, compilar
autoencoder.compile(loss=args.loss_function, optimizer=args.optimizer)

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
history = autoencoder.fit(x_train, x_train,
                          validation_split=args.validation_size,
                          batch_size=args.batch_size,
                          epochs=args.epochs,
                          shuffle=False,
                          verbose=1,
                          callbacks=callbacks)

# ver resutados de entrenamiento
training_history(history, model_name="Autoencoder", filename="Autoencoder")

# sacar las primeras 6 capas la input se cuenta
encoder = tf.keras.models.Sequential(autoencoder.layers[:6])
encoder.summary()

# ver la representación del espacio latente
latent_representation = encoder.predict(x_test)
latent_space_visualization(latent_representation, y_test)

# analizar el error de reconstrucción
x_test_pred = autoencoder.predict(x_test)
mse = np.mean(np.power(x_test - x_test_pred, 2), axis=1)
mse = np.reshape(mse, (-1, 1))
error_df = np.concatenate([mse, y_test], axis=1)
error_df = pd.DataFrame(error_df, columns=['error_reconstruccion', "clase"])

# calculamos desviación absoluta promedio para fijar el thershold
z_scores = mad_score(mse)
thershold = 0.5
outliers = z_scores > thershold

print(f"Detectado {np.sum(outliers):,} outliers en un total de",
      f"{np.size(z_scores):,}",
      f"transacciones [{np.sum(outliers)/np.size(z_scores):.2%}].")

cm = confusion_matrix(y_test, outliers)
(tn, fp, fn, tp) = cm.flatten()
print("Recall:", tp / (tp + fn) * 100)
print("Precision:", tp / (tp + fp) * 100)

recall = recall_score(y_test, outliers, average="binary")
