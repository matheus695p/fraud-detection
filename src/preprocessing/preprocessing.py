

def supervised_preparation(df_features, df_target):
    """
    Hacer la preparación para un aprendizaje supervizado

    Parameters
    ----------
    df_features : datafrane
        df con los features.
    df_target : TYPE
        df con los targets.

    Returns
    -------
    x : numpy.array
        numpy array de los features.
    y : numpy.array
        numpy array de los targets.

    """
    x = df_features.to_numpy()
    y = df_target.to_numpy()
    return x, y


def selection_by_correlation(dataset, threshold=0.8):
    """
    Selecciona solo una de las columnas altamente correlacionadas y elimina
    la otra
    Parameters
    ----------
    dataset : dataframe
        dataset sin la variable objectivo.
    threshold : float
        modulo del valor threshold de correlación pearson.
    Returns
    -------
    dataset : dataframe
        dataset con las columnas eliminadas.
    """
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            cond1 = (corr_matrix.iloc[i, j] >= threshold)
            if cond1 and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname]
    dataset = dataset.reset_index(drop=True)
    return dataset
