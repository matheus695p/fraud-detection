import numpy as np


def mad_score(points):
    """
    Computa mean absolute deviation  de un array de puntos
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm

    Parameters
    ----------
    points : array
        puntos, que en este caso son el error de reconstrucción.

    Returns
    -------
    float
        mad_score.

    """
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)
    return 0.6745 * ad / mad
