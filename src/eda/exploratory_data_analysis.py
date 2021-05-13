import pandas as pd
from pandas_profiling import ProfileReport


def profiling_report(path_csv, minimal_mode=False, dark_mode=True):
    """
    Utiliza la libreria pandas_profiling para hacer una exploración visual
    rápida de los datos

    Parameters
    ----------
    path_csv : string
        path al .csv que contiene la data.
    minimal_mode : string, optional
        En el caso de que sea True, hace cálculo de correlaciones no lineales.
        The default is False.
    dark_mode : string, optional
        si es en el modo oscuro o no. The default is True.

    Returns
    -------
    .html con la exploración de los datos.

    """
    # lectura del .csv
    df = pd.read_csv(path_csv)
    # esto hace la logica de como guardar el archivo nomás
    if dark_mode:
        type_html = "-black"
    else:
        type_html = ""
    if minimal_mode:
        title_mode = "-no expensive computations"
        mode = title_mode.replace(" ", "-")
    else:
        title_mode = ""
        mode = title_mode.replace(" ", "-")

    title = "Análisis Eploratorio de Data: RappiPay"
    prof = ProfileReport(df,
                         title=title,
                         explorative=False,
                         minimal=minimal_mode,
                         orange_mode=dark_mode)
    # guardar el html
    path_output =\
        f'results/exploratory-analysis/{mode}-eda.html'
    prof.to_file(output_file=path_output)
