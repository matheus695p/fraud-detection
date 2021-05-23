import warnings
from argparse import ArgumentParser
warnings.filterwarnings("ignore", category=DeprecationWarning)


def xgboost_arguments():
    """
    El parser de argumentos de parámetros para el xgboost params grid
    Returns
    -------
    args : argparser
        argparser con todos los parámetros del modelo.
    """
    # argumentos
    parser = ArgumentParser()
    # agregar donde correr y guardar datos
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--max_depth', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--early_stopping', type=int, default=20)
    parser.add_argument('--objective', type=str, default="binary:logistic")
    parser.add_argument('--eval_metric', type=str, default="recall")
    parser.add_argument('--n_gpus', type=int, default=1)
    args = parser.parse_args()
    return args


def nn_arguments():
    """
    El parser de argumentos de parámetros que hay que setiar para entrenar
    una red fully connected
    Returns
    -------
    args : argparser
        argparser con todos los parámetros del modelo.
    """
    # argumentos
    parser = ArgumentParser()
    # agregar donde correr y guardar datos
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--min_delta', type=float, default=1e-8)
    parser.add_argument('--lr_factor', type=float, default=0.75)
    parser.add_argument('--lr_patience', type=int, default=15)
    parser.add_argument('--random_state', type=int, default=20)
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--validation_size', type=float, default=0.2)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--metric', type=str, default="recall")
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument(
        '--loss_function', type=str, default="binary_crossentropy")
    args = parser.parse_args()
    return args


def stacked_neural_net_arguments():
    """
    El parser de argumentos de parámetros que hay que setiar para entrenar
    una red red fully connected y convolucional 2D
    Returns
    -------
    args : argparser
        argparser con todos los parámetros del modelo.
    """
    # argumentos
    parser = ArgumentParser()
    # agregar donde correr y guardar datos
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--min_delta', type=float, default=1e-8)
    parser.add_argument('--lr_factor', type=float, default=0.75)
    parser.add_argument('--lr_patience', type=int, default=15)
    parser.add_argument('--random_state', type=int, default=20)
    parser.add_argument('--lr_min', type=float, default=1e-3)
    parser.add_argument('--validation_size', type=float, default=0.2)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--metric', type=str, default="recall")
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument(
        '--loss_function', type=str, default="binary_crossentropy")
    args = parser.parse_args()
    return args


def autoencoder_arguments():
    """
    El parser de argumentos de parámetros que hay que setiar para entrenar
    un autoencoder para reducción de dimensionalidad
    Returns
    -------
    args : argparser
        argparser con todos los parámetros del modelo.
    """
    # argumentos
    parser = ArgumentParser()
    # agregar donde correr y guardar datos
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--min_delta', type=float, default=1e-8)
    parser.add_argument('--lr_factor', type=float, default=0.75)
    parser.add_argument('--lr_patience', type=int, default=15)
    parser.add_argument('--random_state', type=int, default=20)
    parser.add_argument('--lr_min', type=float, default=1e-3)
    parser.add_argument('--validation_size', type=float, default=0.2)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--metric', type=str, default="recall")
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument(
        '--loss_function', type=str, default="mse")
    args = parser.parse_args()
    return args


def vae_arguments():
    """
    El parser de argumentos de parámetros que hay que setiar para entrenar
    un variational autoencoder para hacer aumentación de data

    Returns
    -------
    args : argparser
        argparser con todos los parámetros del modelo.
    """
    # argumentos
    parser = ArgumentParser()
    # agregar donde correr y guardar datos
    parser.add_argument('--original_dim', type=int, default=64)
    parser.add_argument('--intermediate_dim', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--epsilon_std', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--min_delta', type=float, default=1e-8)
    parser.add_argument('--lr_factor', type=float, default=0.75)
    parser.add_argument('--lr_patience', type=int, default=15)
    parser.add_argument('--random_state', type=int, default=20)
    parser.add_argument('--lr_min', type=float, default=1e-3)
    parser.add_argument('--validation_size', type=float, default=0.2)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--loss_function', type=str, default="mse")
    args = parser.parse_args()
    return args
