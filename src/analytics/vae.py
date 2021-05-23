from keras import backend as K
from keras.layers import Layer


def nll(y_true, y_pred):
    """
    Probabilidad logarítmica negativa (Bernoulli).

    Parameters
    ----------
    y_true : numpy.array
        datos reales.
    y_pred : numpy.array
        predicciones.
    Returns
    -------
    tf.tensor
        probabilidad logaritmica negativa.

    """
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """
    Capa de transformación de identidad que agrega divergencia KL
    hasta la pérdida final del modelo.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs
