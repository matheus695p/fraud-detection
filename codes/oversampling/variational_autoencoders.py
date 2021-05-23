import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K
from src.analytics.vae import (KLDivergenceLayer, nll)
from src.utils.visualizations import (training_history, vae_latent_space)
from src.config.config import vae_arguments

# parametros del variational autoencoder
args = vae_arguments()

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

# división del conjutno de datos
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=args.validation_size, stratify=y,
    random_state=args.random_state)

# creación del decoder
decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(
        args.intermediate_dim, input_dim=args.latent_dim, activation='relu'),
    tf.keras.layers.Dense(args.original_dim, activation='sigmoid')])

# construción del variational autoencoder
x = tf.keras.layers.Input(shape=(args.original_dim,))
h = tf.keras.layers.Dense(args.intermediate_dim, activation='relu')(x)

# espacio latente
z_mu = tf.keras.layers.Dense(args.latent_dim)(h)
z_log_var = tf.keras.layers.Dense(args.latent_dim)(h)

# aplicar divergencia KL
z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = tf.keras.layers.Lambda(lambda t: K.exp(.5*t))(z_log_var)
# generar la distribución
eps = tf.keras.layers.Input(tensor=K.random_normal(
    stddev=args.epsilon_std, shape=(K.shape(x)[0], args.latent_dim)))
# sacar el espacio latente
z_eps = tf.keras.layers.Multiply()([z_sigma, eps])
z = tf.keras.layers.Add()([z_mu, z_eps])
x_pred = decoder(z)

# compilar el grafo estatico
vae = tf.keras.models.Model(inputs=[x, eps], outputs=x_pred)
vae.compile(optimizer=args.optimizer, loss=nll)

# resumen
print(vae.summary())

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

# entrenar el vae
history = vae.fit(x_train, x_train,
                  validation_split=0.2,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  shuffle=False,
                  verbose=1,
                  callbacks=callbacks)

# ver resutados de entrenamiento
training_history(history, model_name="VAE", filename="VAE")

# encoder
encoder = tf.keras.models.Model(x, z_mu)

# hacer la representación del conjunto de test
z_test = encoder.predict(x_test, batch_size=args.batch_size)

# plot del espacio latente
vae_latent_space(z_test, y_test)
