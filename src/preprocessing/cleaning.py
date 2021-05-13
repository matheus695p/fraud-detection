import numpy as np


def renaming_columns(df):
    """
    Convertir columnas de un dataframe en flotantes
    Parameters
    ----------
    df : dataframe
        dataframe que se deben renombrar las columnas.
    Returns
    -------
    df : dataframe
        dataframe con las columnas renombradas.
    """
    for col in df.columns:
        try:
            new_col = col.lower()
            df.rename(columns={col: new_col}, inplace=True)
        except Exception:
            pass
    df.reset_index(drop=True, inplace=True)
    return df


def convert_df_float(df):
    """
    Convertir columnas de un dataframe en flotantes
    Parameters
    ----------
    df : dataframe
        pasar todas las columnas de un dataset tabular a float.
    Returns
    -------
    df : TYPE
        DESCRIPTION.
    """
    for col in df.columns:
        try:
            df[col] = df[col].apply(float)
        except Exception:
            pass
    df.reset_index(drop=True, inplace=True)
    return df


def drop_spaces_data(df):
    """
    sacar los espacios de columnas que podrián venir interferidas
    Parameters
    ----------
    df : dataframe
        input data
    column : string
        string sin espacios en sus columnas
    Returns
    -------
    """
    for column in df.columns:
        try:
            df[column] = df[column].str.lstrip()
            df[column] = df[column].str.rstrip()
        except Exception as e:
            print(e)
            pass
    return df


def make_empty_identifiable(value):
    """
    Parameters
    ----------
    value : int, string, etc
        valor con el que se trabaja.
    Returns
    -------
    nans en los vacios.
    """
    if value == "":
        output = np.nan
    else:
        output = value
    return output


def replace_empty_nans(df):
    """
    Parameters
    ----------
    df : int, string, etc
        valor con el que se trabaja.
    Returns
    -------
    nans en los vacios.
    """
    for col in df.columns:
        print("buscando vacios en:", col, "...")
        df[col] = df[col].apply(lambda x: make_empty_identifiable(x))
    return df
