import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm as cm
warnings.filterwarnings("ignore")
plt.style.use('dark_background')


def kernel_density_estimation(df, col, name="Colum", bw_adjust=0.1):
    """
    Estimación de la función densidad de prob
    Parameters
    ----------
    df : dataframe
        dataframe a realizar el pairplot.
    col : string
        nombre de la columna a realizar el violinplot.
    name : string, optional
        nombre del gráfico. The default is "Entrenamiento".
    bw_adjust : float, optional
        Ajuste de la distribución. The default is 0.1.
    Returns
    -------
    Estimación de la distribución de la columna.
    """
    plt.style.use('dark_background')
    sns.set(font_scale=1.5)
    pplot = sns.displot(df, x=col, kind="kde", bw_adjust=bw_adjust)
    pplot.fig.set_figwidth(10)
    pplot.fig.set_figheight(8)
    pplot.set(title=name)


def piechart_plot(df, col="class"):
    """
    Pie chart de la clase que estamos tratando de predecir, para clarificar
    el desbalance de la data en esos casos

    Parameters
    ----------
    df : dataframe
        base de datos de transacciones bancarias.
    col : string, optional
        nombre de la columna, debe ser categorica y binaria.
        The default is "class".

    Returns
    -------
    Pie chart con el plot de los datos.

    """
    mpl.rcParams['font.size'] = 25
    description = pd.DataFrame(df[col].value_counts())
    description.reset_index(drop=False, inplace=True)
    description.columns = ["clase", "conteo"]
    description["porcentaje"] = description["conteo"] / \
        description["conteo"].sum() * 100
    explode = list(description["clase"])
    labels = ["Transacción legitima", "Fraude"]
    sizes = list(description["porcentaje"])
    fig, ax = plt.subplots(1, figsize=(22, 12))
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal')
    plt.title("Pie chart Desbalance Clases", fontsize=25)


def pairplot_df(df, col1, col2, path="images/results/pairplot"):
    """
    Empezar a hacer pairplots para ver correlaciones entre las variables

    Parameters
    ----------
    df : dataframe
        ads.
    col1 : string
        nombre de la columna 1.
    col2 : string
        nombre de la columna 2.
    path : string, optional
        path de guardado de la imagen.
        The default is "images/results/pairplot".

    Returns
    -------
    Plot de la correlación entre dos variables.

    """
    plt.style.use('dark_background')
    # tamaño de la letra
    letter_size = 20
    # caracteristicas del dataset
    fig, ax = plt.subplots(1, figsize=(22, 12))
    plt.scatter(df[[col1]].to_numpy(), df[[col2]].to_numpy(),
                color='orangered')
    # filtro de posibilidad
    plt.ylim(df[col1].quantile(0.02), df[col1].quantile(0.98))
    plt.xlim(df[col2].quantile(0.02), df[col2].quantile(0.98))
    # plt.ylim(0, df[col1].quantile(0.99))
    # plt.xlim(0, df[col2].quantile(0.99))
    # restricciones
    col1 = col1.replace("_", " ").title()
    col2 = col2.replace("_", " ").title()
    titulo = f"{col1} vs {col2}"
    plt.title(titulo, fontsize=30)
    plt.xlabel(f'{col1}', fontsize=30)
    plt.ylabel(f'{col2}', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=22)
    plt.legend([f'{col1} vs {col2}'], loc='upper left',
               prop={'size': letter_size+5})
    plt.show()
    path = path + f"/{col1}_vs_{col2}.png"
    fig.savefig(path)


def plot_sequence(df, col):
    """
    Hacer plot de la evolución en función del tiempo de una metrica en
    particular

    Parameters
    ----------
    df : dataframe
        ads ya trabajado.
    col : string
        nombre de columna.

    Returns
    -------
    Evolución de la columna en función del tiempo.

    """
    indices = [i+1 for i in range(len(df))]
    letter_size = 20
    q1_ = 0.1
    q2_ = 0.9
    q1 = df[col].quantile(q1_)
    q2 = df[col].quantile(q2_)
    metric = df[(df[col] >= q1) & (df[col] <= q2)][col]
    mean = str(round(metric.mean(), 3))
    phrase = "El promedio es:" + f"{mean}"
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, figsize=(20, 12))
    ax.plot(indices, list(df[col]), 'gold', linewidth=2)
    ax.set_xlabel('Tiempo', fontname="Arial", fontsize=letter_size)
    ax.set_ylabel(f'{col}', fontname="Arial", fontsize=letter_size+2)
    ax.set_title(f'Evolución en función del tiempo: {col}' + '\n' + phrase,
                 fontname="Arial", fontsize=letter_size+10)
    ax.legend([col], loc='upper left',
              prop={'size': letter_size+5})
    plt.ylim(df[col].quantile(q1_), df[col].quantile(q2_))
    plt.xlim(df[col].quantile(q1_), df[col].quantile(q2_))
    # Tamaño de los ejes
    for tick in ax.get_xticklabels():
        tick.set_fontsize(letter_size)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(letter_size)
    plt.xticks(rotation=75)
    plt.show()


def correlation_matrix(df, method="pearson"):
    """
    Hacer matriz de correlaciones según distintos métodos de correlación,
    para analizar a simple vista los datos

    Parameters
    ----------
    df : dataframe
        dataset que estamos analizando.
    method : string, optional
        método de correlación soportados por libreria pandas
        {‘pearson’, ‘kendall’, ‘spearman’} .
        The default is "pearson".

    Returns
    -------
    None.

    """
    letter_size = 25
    fig = plt.figure(figsize=(22, 12))
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap('hot_r', 30)
    ax = fig.add_subplot(111)
    size = int(len(list(df.columns))/2)
    corr = df.corr(method=method)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap=cmap)
    plt.xticks(range(len(corr.columns)), corr.columns, fontsize=letter_size)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=letter_size)
    ax.set_xticklabels(df.columns, fontsize=letter_size)
    ax.set_yticklabels(df.columns, fontsize=letter_size)
    plt.xticks(rotation=90)
    # plt.yticks(rotation=90)
    ax.set_title(f"Matriz de correlación, método: {method}",
                 fontname="Arial", fontsize=letter_size+10)
    path = f"results/correlations/{method}.png"
    fig.savefig(path)


def get_fraude_distribution(obj):
    """
    Visualizar la distribución de las clases en la clasificación

    Parameters
    ----------
    obj : obj
        targets.
    Returns
    -------
    count_dict : dict
        diccionario para ver las distribuciones.
    """
    count_dict = {
        "legitimo": 0,
        "fraude": 0,
    }

    for i in obj:
        if i == 0:
            count_dict['legitimo'] += 1
        elif i == 1:
            count_dict['fraude'] += 1
        else:
            print("Check classes.")
    return count_dict


def watch_classification_distribution(y_train, y_test):
    """
    Ver las distribuciones de los targets
    Parameters
    ----------
    y_train : numpy array
        target separado de entrenamiento y validación.
    y_test : numpy array
        target separado de testeo.
    Returns
    -------
    Verficar si las distribuciones fueron bien hechas.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 7))
    # entrenamiento
    data = pd.DataFrame.from_dict([get_fraude_distribution(y_train)])
    relation_train = (100 * data["fraude"] /
                      (data["legitimo"] + data["fraude"])).mean()
    relation_train = str(round(relation_train, 5))

    data = pd.DataFrame.from_dict([get_fraude_distribution(y_test)])
    relation_test = (100 * data["fraude"] /
                     (data["legitimo"] + data["fraude"])).mean()
    relation_test = str(round(relation_test, 5))

    title_train =\
        "Distribución en Train Set" + "\n" + f"relación [%]: {relation_train}"
    sns.barplot(data=pd.DataFrame.from_dict([
        get_fraude_distribution(y_train)]).melt(),
        x="variable", y="value", hue="variable", color="red",
        ax=axes[0]).set_title(title_train)

    title_test =\
        "Distribución en Test Set" + "\n" + f"relación [%]: {relation_test}"
    # validación
    sns.barplot(data=pd.DataFrame.from_dict([
        get_fraude_distribution(y_test)]).melt(),
        x="variable", y="value", hue="variable", color="blue",
        ax=axes[1]).set_title(title_test)


def plot_instance_training(history, epocas_hacia_atras, model_name,
                           filename):
    """
    Sacar el historial de entrenamiento de epocas en partivular
    Parameters
    ----------
    history : object
        DESCRIPTION.
    epocas_hacia_atras : int
        epocas hacia atrás que queremos ver en el entrenamiento.
    model_name : string
        nombre del modelo.
    filename : string
        nombre del archivo.
    Returns
    -------
    bool
        gráficas de lo ocurrido durante el entrenamiento.
    """
    plt.style.use('dark_background')
    letter_size = 20
    # Hist training
    largo = len(history.history['loss'])
    x_labels = np.arange(largo-epocas_hacia_atras, largo)
    x_labels = list(x_labels)
    # Funciones de costo
    loss_training = history.history['loss'][-epocas_hacia_atras:]
    loss_validation = history.history['val_loss'][-epocas_hacia_atras:]
    # Figura
    fig, ax = plt.subplots(1, figsize=(16, 8))
    ax.plot(x_labels, loss_training, 'gold', linewidth=2)
    ax.plot(x_labels, loss_validation, 'r', linewidth=2)
    ax.set_xlabel('Epocas', fontname="Arial", fontsize=letter_size-5)
    ax.set_ylabel('Función de costos', fontname="Arial",
                  fontsize=letter_size-5)
    ax.set_title(f"{model_name}", fontname="Arial", fontsize=letter_size)
    ax.legend(['Entrenamiento', 'Validación'], loc='upper left',
              prop={'size': letter_size-5})
    # Tamaño de los ejes
    for tick in ax.get_xticklabels():
        tick.set_fontsize(letter_size-5)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(letter_size-5)
    plt.show()
    return fig


def training_history(history, model_name="NN", filename="NN"):
    """
    Según el historial de entrenamiento que hubo plotear el historial
    hacía atrás de las variables
    Parameters
    ----------
    history : list
        lista con errores de validación y training.
    model_name : string, optional
        nombre del modelo. The default is "Celdas LSTM".
    filename : string, optional
        nombre del archivo. The default is "LSTM".
    Returns
    -------
    None.
    """
    size_training = len(history.history['val_loss'])
    fig = plot_instance_training(history, size_training, model_name,
                                 filename + "_ultimas:" +
                                 str(size_training) + "epocas")

    fig = plot_instance_training(history, int(1.5 * size_training / 2),
                                 model_name,
                                 filename + "_ultimas:" +
                                 str(1.5 * size_training / 2) + "epocas")
    # guardar el resultado de entrenamiento de la lstm
    fig.savefig(f"results/models/{model_name}_training.png")

    fig = plot_instance_training(history, int(size_training / 2),
                                 model_name,
                                 filename + "_ultimas:" + str(
                                     size_training / 2) + "epocas")

    fig = plot_instance_training(history, int(size_training / 3), model_name,
                                 filename + "_ultimas:" +
                                 str(size_training / 3) + "epocas")
    fig = plot_instance_training(history, int(size_training / 4), model_name,
                                 filename + "_ultimas:" + str(
                                     size_training / 4) + "epocas")
    print(fig)


def plot_confusion_matrix(df_confusion, title='Matriz de confusion',
                          cmap=plt.cm.hot):
    """
    Visualizar la matriz de confusión de la clasificación
    Parameters
    ----------
    df_confusion : dataframe
        matriz de confusión.
    title : string, optional
        titulo del gráficoo. The default is 'Matriz de confusion'.
    cmap : TYPE, optional
        DESCRIPTION. The default is plt.cm.gray_r.
    Returns
    -------
    Plot de la matriz de confusión.
    """
    plt.matshow(df_confusion, cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    # plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
