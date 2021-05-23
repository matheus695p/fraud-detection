![Build Status](https://www.repostatus.org/badges/latest/concept.svg)

# fraud-detection


## Organización del repositorio

Repo donde se hace el análisis de detección de fraude para Rappi, los códigos están ordenados de acuerdo al siguiente arbol.

```sh
├───codes      ---> Carpeta códigos usados para hacer el proyecto
│   ├───eda                  ---> Carpeta donde están los scripts del análisis exploratorio realizado al dataset
│   ├───oversampling         ---> Carpeta donde están los scripts del proceso de sobre muestreo realizado al dataset para obtener más datos
│   ├───predictors           ---> Carpeta donde están los scripts de todos los predictores
│   └───preprocessing        ---> Carpeta donde están los scripts de todo el proceso de feature engenieering realizado al dataset
├───data                     ---> Carpeta donde siendo guardados los datos que van pasando de un proceso a otro
├───results    ---> Carpeta donde siendo guardados imagenes que estan saliendo de los procesos
│   ├───correlations         ---> Carpeta donde se guardan algunas correlaciones lineales y no lineales de los datos
│   ├───exploratory-analysis ---> Carpeta donde se guardan los .html y .png de los resultados exploratorios de los datos
│   ├───imbalance-target     ---> Carpeta donde se muestra desbalance de la data
│   ├───models               ---> Carpeta donde se guardan resultados de entrenamientos de modelos
│   └───scatter-plot         ---> Carpeta donde se hacen plots scatter para ver tendencias entre las variables
└───src       ---> Carpeta de modulos (clases y funciones del proyecto)
    ├───analytics            ---> Módulo analitico donde se encuetran funciones para transformaciones, metricas y modelos           
    ├───config               ---> Módulo de configuraciones de scripts, donde se guardan hiperparametros de modelos utilizados
    ├───eda                  ---> Módulo para hacer análisis exploratorios de datos
    ├───preprocessing        ---> Módulo para hacer preprocesamiento de datos
    └───utils                ---> Módulo para guardar funciones útilies y visualizaciones
```
## Librerias necearias

```sh
$ git clone https://github.com/matheus695p/fraud-detection.git
$ cd fraud-detection
$ pip install -r requirements.txt

El requirements fue sacado de un 

$ pip freeze > requirements.txt

A un ambiente con el que trabajé por lo tanto muestra todos los paquetes instalados en ese ambiente, algunos por ser por conda, pueden venir con rutas locales,
en el caso de que fallé hacer la siguiente lista de comandos en el primer ambiente que tendrá tensorflow


$ pip install pandas
$ pip install pandas-profiling
$ pip install scikit-learn
$ pip install statsmodels
$ pip install matplotlib
$ pip install seaborn
$ pip install imblearn
$ pip install xgboost
$ pip install tensorflow-gpu==2.4.0  [esta versión me permite traerme cuda v11 que es lo que necesita mi GPU como driver]
$ pip install autoplotter

Para el segundo ambiente, ocuparemos por debajo pytorch para entrenar las redes generativas adversarias a través de tabgan
este problema ocurre debido a las versiones de cuda incompatibles entre tensorflow y pytorch, si quieres entrenar todo en CPU
podrías juntar ambas listas de paquetes, pero no lo recomiendo por la velocidad de entrenamiento en GPU es un 20 X.

$ pip install pandas
$ pip install tabgan
$ pip install scikit-learn

```
# Proceso de limpieza de los datos

El proceso de limpieza busca ordenar, hacer visibles los nans en el dataframe convertir a float las columnas que sean numericas y renombrar las columnas
para que estas estén en mimusculas y sin espacios con el fin de evitar futuros problemas con conexiones a bases de datos.


```zh
├───codes  
    └───preprocessing
            cleaning.py    ---> Spript de limpieza de los datos
```


# EDA (Análisis exploratorio de los datos)

## Desbalance del target del problema


<p align="center">
  <img src="./results/imbalance-target/pie_chart.png">
</p>


## Correlación de variables

El coeficiente de correlación de Pearson (r) es una medida de correlación lineal entre dos variables. Su valor se encuentra entre -1 y +1, -1 indica una correlación lineal negativa total, 0 indica que no hay correlación lineal y 1 indica una correlación lineal positiva total. Además, r es invariante bajo cambios separados en la ubicación y escala de las dos variables, lo que implica que para una función lineal el ángulo con el eje x no afecta a r.

Para calcular r para dos variables X e Y, se divide la covarianza de X e Y por el producto de sus desviaciones estándar.
El coeficiente de correlación de rango de Spearman (ρ) es una medida de correlación monótona entre dos variables y, por lo tanto, es mejor para detectar correlaciones monotónicas 
no lineales que la r de Pearson. Su valor se encuentra entre -1 y +1, -1 indica una correlación monótona negativa total, 0 indica que no hay correlación monótona y 1 indica una correlación monótona positiva total. Para calcular ρ para dos variables X e Y, se divide la covarianza de las variables de rango de X e Y por el producto de sus desviaciones estándar.

De manera similar al coeficiente de correlación de rangos de Spearman, el coeficiente de correlación de rangos de Kendall (τ) mide la asociación ordinal entre dos variables. Su valor se encuentra entre -1 y +1, -1 indica una correlación negativa total, 0 indica que no hay correlación y 1 indica una correlación positiva total. Para calcular τ para dos variables X e Y, se determina el número de pares de observaciones concordantes y discordantes. τ viene dado por el número de pares concordantes menos los pares discordantes dividido por el número total de pares.



<p align="center">
  <img src="./results/correlations/pearson.png">
</p>


<p align="center">
  <img src="./results/correlations/spearman.png">
</p>


<p align="center">
  <img src="./results/correlations/kendall.png">
</p>


## Análisis exploratorio completo

Dado que el análisis exploratorio por si solo no aporta valor a la solución del problema, decidí apoyarme en dos librerías que facilitan mucho este análisis, una es pandas-profiling y la otra autoplotter que permiten hacer visualizaciones muy rápidas de los datos logrando visualizar distribuciones, tipos de datos, datos faltantes, correlaciones, etc. Los códigos de estos análisis están en:


```zh
├───eda
│       eda.py           ----> Análisis exploratorio con pandas profilling y librerias de visualización como matplotlib y seaborn
│       autoploter.py    ----> Lanzar app para poder visualizar todo tipo de gráficos entre las variables en el explorador, sacar el puerto http://127.0.0.1:5000/ para la visualización

├───results
│   ├───exploratory-analysis
│   │       expensive-computations-eda.html     ----> .html con el EDA de los datos limpios 
│   │       no-expensive-computations-eda.html  ----> .html con el EDA de los datos limpios

```




# Feature Engineering
