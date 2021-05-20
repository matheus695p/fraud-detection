import os
import math
import pandas as pd
import numpy as np


def try_create_folder(path="images"):
    """
    Intentar crear carpeta
    Parameters
    ----------
    path : string
        direccion.
    """
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)


def get_prime_factors(number):
    """
    Obtiene lista con los numeros primos de la factorización de un número
    nos sirve para encontrar el tamaño del reshape para la cnn
    Parameters
    ----------
    number : int
        numero a factorizar.
    Returns
    -------
    prime_factors : list
        lista con la factorización prima.
    """
    prime_factors = []
    while number % 2 == 0:
        prime_factors.append(2)
        number = number / 2
    for i in range(3, int(math.sqrt(number)) + 1, 2):
        while number % i == 0:
            prime_factors.append(int(i))
            number = number / i
    if number > 2:
        prime_factors.append(int(number))

    return prime_factors


def multiple_items(listt):
    """
    Pone el valor de los items
    Parameters
    ----------
    listt : TYPE
        DESCRIPTION.
    Returns
    -------
    value : TYPE
        DESCRIPTION.
    """
    if len(listt) == 0:
        value = -1
    else:
        value = 1
        for item in listt:
            value = value * item
    return value


def input_shape(prime_factorization, natural_shape=-1):
    """
    Dada la factorización de números primos, encuentra el tamaño de la
    matriz que hay que hacer el reshape para entrenar la red
    Parameters
    ----------
    prime_factorization : list
        factorización prima.
    natural_shape : TYPE, optional
        DESCRIPTION. The default is -1.
    Returns
    -------
    shape : TYPE
        DESCRIPTION.
    """
    len_ = len(prime_factorization)

    root = np.sqrt(natural_shape)
    if root - int(root) == 0:
        print("Factorización exacta")
        shape = (int(root), int(root))
    else:
        dimensions = []
        if len_ >= 2:
            for j in range(0, len_):
                # buscar las posibilidades
                first_shape = prime_factorization[0: j]
                second_shape = prime_factorization[j:]
                # dimensiones posibles
                first_dim = multiple_items(first_shape)
                second_dim = multiple_items(second_shape)
                print("Dimensiones posibles:", "(", first_dim, ",",
                      second_dim, ")")
                dimensions.append([first_dim, second_dim])
            dimensions = pd.DataFrame(dimensions, columns=["first", "second"])
            dimensions["diff"] = np.abs(
                dimensions["first"] - dimensions["second"])
            dimensions.replace(-1, 0, inplace=True)
            minimum = dimensions["diff"].min()
            dimensions.replace(0, -1, inplace=True)
            dimensions = dimensions[dimensions["diff"] == minimum]
            dimensions.reset_index(drop=True, inplace=True)
            shape = (dimensions["second"][0], dimensions["first"][0])

        else:
            print("No es posible entregar una factorización de este número")
            shape = (-1, natural_shape)
    return shape
