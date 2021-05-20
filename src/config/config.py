import warnings
from argparse import ArgumentParser
warnings.filterwarnings("ignore", category=DeprecationWarning)


def nn_arguments():
    """
    El parser de argumentos de parámetros que hay que setiar para entrenar
    una red deep renewal
    Returns
    -------
    args : argparser
        argparser con todos los parámetros del modelo.
    """
    # argumentos
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--fff", help="haciendo cosas de python", default="1")
    # agregar donde correr y guardar datos
    parser.add_argument('--batch_size', type=int, default=2048)
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
    una red deep renewal
    Returns
    -------
    args : argparser
        argparser con todos los parámetros del modelo.
    """
    # argumentos
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--fff", help="haciendo cosas de python", default="1")
    # agregar donde correr y guardar datos
    parser.add_argument('--batch_size', type=int, default=1024)
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
