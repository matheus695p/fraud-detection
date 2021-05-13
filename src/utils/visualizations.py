import warnings
import pandas as pd
import seaborn as sns
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


def pichart_plot(df, col="class"):
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
    plt.title("Pie chart Desbalance Clases", fontsize=30)


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

    # plt.scatter(df[[col1]].to_numpy(), df[[col1]].to_numpy(),
    #             color='blue')
    # filtro de posibilidad
    plt.ylim(df[col1].quantile(0.1), df[col1].quantile(0.9))
    plt.xlim(df[col2].quantile(0.1), df[col2].quantile(0.9))
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
    # fechas = list(df["fecha_ini_turno"].apply(lambda x: x[0:10]))
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


def correlation_matrix(df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Abalone Feature Correlation')
    labels = list(df.columns)
    ax1.set_xticklabels(labels, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    fig.colorbar(cax, ticks=[.75, .8, .85, .90, .95, 1])
    plt.show()
    plt.show()
