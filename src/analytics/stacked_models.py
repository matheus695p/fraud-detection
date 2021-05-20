import tensorflow as tf


def create_nn(dim):
    """
    Crear modelo de la red neuronal
    Parameters
    ----------
    dim : int
        features que hay en la matriz de entrada.
    Returns
    -------
    model : tf.model
        modelo de la red fully connected.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_dim=dim))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    return model


def create_lstm(input_shape):
    """
    Crear modelo lstm
    Parameters
    ----------
    input_shape : array
         matriz espacio temporal.
    Returns
    -------
    model : tf.model
        modelo de la red lstm.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=512,
                                   input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(0.2))
    return model


def create_cnn1d(input_shape_):
    """
    Crear modelo de una red convolucional en 1 dimensión
    Parameters
    ----------
    input_shape_ : (features, 1)
        tamaño de la matriz espacial.
    Returns
    -------
    model : tf.model
        red convolucional.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                                     activation='relu',
              input_shape=input_shape_))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv1D(
        filters=128, kernel_size=5, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(
        filters=256, kernel_size=10, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    return model


def create_cnn2d(input_shape_):
    """
    Crear red convolucional 2D
    Parameters
    ----------
    input_shape_ : array
        features reshape, para tomar en cuenta kernels espacio temporales.
    Returns
    -------
    cnn : tf.model
        red neuronal convolicional 2D.
    """
    cnn = tf.keras.Sequential()
    cnn.add(tf.keras.layers.Conv2D(16, input_shape=input_shape_,
                                   kernel_size=(3, 3), padding="same",
                                   activation="relu"))
    cnn.add(tf.keras.layers.Dropout(0.2))
    cnn.add((tf.keras.layers.Conv2D(32, (2, 2), padding="same",
                                    activation="relu")))
    # 1 capa de máx pooling
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn.add(tf.keras.layers.Dropout(0.2))
    cnn.add((tf.keras.layers.Conv2D(64, (2, 1), padding="same",
                                    activation="relu")))
    cnn.add(tf.keras.layers.Dropout(0.2))
    cnn.add((tf.keras.layers.Conv2D(128, (1, 2), padding="same",
                                    activation="relu")))
    cnn.add(tf.keras.layers.Dropout(0.2))
    # cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 3)))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(128))
    cnn.add(tf.keras.layers.BatchNormalization())
    cnn.add(tf.keras.layers.Activation("relu"))
    cnn.add(tf.keras.layers.Dropout(0.3))
    return cnn


def create_conv_lstm(input_size):
    """
    Crear red neuronal conv_lstm
    Parameters
    ----------
    input_size : array
        matriz espacio temporal, con timesteps.
    Returns
    -------
    model : tf.model
        modelo conv lstm.
    """
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(2, 2),
                                         input_shape=input_size,
                                         padding='same',
                                         return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ConvLSTM2D(filters=32, kernel_size=(2, 2),
                                         padding='same',
                                         return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(2, 2),
                                         padding='same',
                                         return_sequences=True))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())

    return model
