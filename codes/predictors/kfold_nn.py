import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_auc_score, recall_score, roc_curve,
                             confusion_matrix, classification_report)
from src.preprocessing.preprocessing import supervised_preparation
from src.utils.visualizations import (watch_classification_distribution,
                                      training_history)
from src.config.config import nn_arguments

# cargar argumentos de la red
args = nn_arguments()
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

# inputs y targets
inputs = x.copy().reset_index(drop=True)
targets = y.copy().reset_index(drop=True)

# kfold
kfold = KFold(n_splits=args.num_folds, shuffle=True)

# dataframe de los resultados
iteration = 1
results = pd.DataFrame()
for train, test in kfold.split(inputs, targets):
    print("Largo de indices de entrenamiento", train.shape, test.shape)
    # Inputs
    x_train = inputs.iloc[train]
    x_test = inputs.iloc[test]

    y_train = targets.iloc[train]
    y_test = targets.iloc[test]

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

    # penalization
    weights = {0: 1, 1: 588}

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
                     validation_split=args.validation_size,
                     batch_size=args.batch_size,
                     epochs=args.epochs,
                     shuffle=False,
                     verbose=1,
                     callbacks=callbacks,
                     class_weight=weights)

    # ver resutados de entrenamiento
    training_history(history, model_name="NN", filename="NN")

    # buscar el mejor thesh
    y_pred = nn.predict(x_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_pred)
    # calcular g-mean para cada thershold
    gmeans = np.sqrt(tpr * (1-fpr))
    # localizar indices de los máximos gmen
    ix = np.argmax(gmeans)
    print(f'Mejor theshold={thresholds[ix]}, G-Mean={gmeans[ix]}')

    # ese thesh falta hacerlo solo x_val
    # asignar thesh
    thresh = thresholds[ix]
    y_pred = nn.predict(x_test)
    y_pred = pd.DataFrame(y_pred, columns=["pred"])
    y_pred["pred"] = y_pred["pred"].apply(lambda x: 1 if x > thresh else 0)
    y_pred = y_pred.to_numpy()
    recall = recall_score(y_test, y_pred.round(), average="binary")
    print(thresh, recall)

    # resultados
    rest = pd.DataFrame([iteration, thresh, recall],
                        columns=["iteracion", "threshold", "recall"])
    # concatenar resultados
    results = pd.concat([results, rest], axis=0)
    iteration += 1

# reset index
results.reset_index(drop=True, inplace=True)

# resultados finales
recall = results["recall"].mean() * 100

print("El recall promedio:", round(recall, 4), "[%]")
