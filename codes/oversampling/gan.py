# import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabgan.sampler import GANGenerator
# from tabgan.sampler import OriginalGenerator, GANGenerator

# cargar datos originales
path = "data/creditcard_featured.csv"
df = pd.read_csv(path)

# separar los datos de fraude para usarlos en el conjunto de test
fraud = df[df["class"] == 1]
fraud.reset_index(drop=True, inplace=True)

# cargar datos oversampling con smote
path = "data/creditcard_smote_oversampling.csv"
over = pd.read_csv(path)

# ordenar para no tener problemas de data drift en el entrenamiento
df = pd.concat([df, over], axis=0)
df = df.sample(frac=1)

# # targets y features
targets = ["class"]
features = list(df.columns)
for tar in targets:
    features.remove(tar)
x = df[features]
y = df[targets]

# división del conjutno de datos
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y,
    random_state=20)

# sacar los datos de test que estaban en el original y agregarlos al test
x_orig = fraud[features]
# concatenar los datos para que la gan intente generar ejemplos parecidos a
# x_test
x_test = pd.concat([x_test, x_orig], axis=0)

# generador
x_gen, y_gen = GANGenerator(
    gen_x_times=1.1, cat_cols=None, bot_filter_quantile=0.001,
    top_filter_quantile=0.999,
    is_post_process=True,
    adversaial_model_params={
        "metrics": "AUC", "max_depth": 2,
        "max_bin": 100, "n_estimators": 500,
        "learning_rate": 0.02, "random_state": 42,
    }, pregeneration_frac=2,
    epochs=500).generate_data_pipe(x_train, y_train,
                                   x_test, deep_copy=True,
                                   only_adversarial=False,
                                   use_adversarial=True)

# concatenacion y limpieza de duplicados que hayan salido
data_gan = pd.concat([x_gen, y_gen], axis=1)
data_over = pd.concat([data_gan, over], axis=0)
data_over = data_over.sample(frac=1)
data_over.drop_duplicates(inplace=True)
data_over.reset_index(drop=True, inplace=True)
# guardar toda la data oversampling
path = "data/creditcard_gan_oversampling.csv"
data_over.to_csv(path, index=False)
