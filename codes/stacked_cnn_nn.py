import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (recall_score, roc_curve)
from src.preprocessing.preprocessing import supervised_preparation
from src.analytics.stacked_models import (create_nn, create_cnn2d)
from src.utils.visualizations import (watch_classification_distribution,
                                      training_history)
from src.utils.utils import get_prime_factors, input_shape
from src.config.config import stacked_neural_net_arguments

# cargar argumentos de la red
args = stacked_neural_net_arguments()
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

watch_classification_distribution(y_train, y_test)

# trabajar con numpy array
x_train_nn, y_train = supervised_preparation(x_train, y_train)
x_test_nn, y_test = supervised_preparation(x_test, y_test)

# Normalización de los datos
sc = MinMaxScaler(feature_range=(0, 1))
# training
x_train = sc.fit_transform(x_train)
# testing
x_test = sc.transform(x_test)

# numero de features
n_features = int(x_train_nn.shape[1])
# función para encontrar la factorización prima de los features, para
# introducir redes conv2D
prime_factorization = get_prime_factors(n_features)
shape = input_shape(prime_factorization,
                    natural_shape=x_train_nn.shape[1])

# se transforma la imagen en tensor 3D para entrenar la red, con 1 canal
x_train_cnn = np.reshape(x_train_nn, (-1, shape[0], shape[1], 1))
x_test_cnn = np.reshape(x_test_nn, (-1, shape[0], shape[1], 1))
# input shape para crear el modelo
cnn_input_shape = x_train_cnn.shape[1:]

# crear red neuronal normal
nn = create_nn(x_train_nn.shape[1])
cnn = create_cnn2d(cnn_input_shape)

# combinar las salidas
combined_input = tf.keras.layers.concatenate([nn.output, cnn.output])

# continuar la concatenación de la red
x = tf.keras.layers.Dense(256)(combined_input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Dense(128)(x)
x = tf.keras.layers.Dense(y_train.shape[1], activation="sigmoid")(x)

# crear el modelo con la extracción de caracteristicas convolucionales
stacked_model = tf.keras.models.Model(
    inputs=[nn.input, cnn.input], outputs=x)
print(stacked_model.summary())

# compilar [grafos estaticos de entrenamienot UWU pytorch > tf]
stacked_model.compile(loss=args.loss_function, optimizer=args.optimizer)

# penalization
weights = {0: 1, 1: 600}

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
history = stacked_model.fit(x=[x_train_nn, x_train_cnn], y=y_train,
                            validation_split=args.validation_size,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            shuffle=False,
                            verbose=1,
                            callbacks=callbacks,
                            class_weight=weights)

# ver resutados de entrenamiento
training_history(history, model_name="stacked_model_NN",
                 filename="stacked_model_NN")

# hacer la prediccion
y_pred = stacked_model.predict(x=[x_train_nn, x_train_cnn])

fpr, tpr, thresholds = roc_curve(y_train, y_pred)
# calcular g-mean para cada thershold
gmeans = np.sqrt(tpr * (1-fpr))
# localizar indices de los máximos gmen
ix = np.argmax(gmeans)
print(f'Mejor theshold={thresholds[ix]}, G-Mean={gmeans[ix]}')

# buscar el mejor thesh
thresh = thresholds[ix]
y_pred = stacked_model.predict(x=[x_test_nn, x_test_cnn])
y_pred = pd.DataFrame(y_pred, columns=["pred"])
y_pred["pred"] = y_pred["pred"].apply(lambda x: 1 if x > thresh else 0)
y_pred = y_pred.to_numpy()
recall = recall_score(y_test, y_pred.round(), average="binary")
print(thresh, recall)
