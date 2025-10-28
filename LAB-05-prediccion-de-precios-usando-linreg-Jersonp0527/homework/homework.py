#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#


import os
import gzip
import json
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression,f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score


# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#


def limpiarDatos(df: pd.DataFrame):
    df = df.copy()
    df["Age"] = 2021 - df["Year"] 
    df = df.drop(columns=["Year","Car_Name"])
    x, y = df.drop(columns=["Present_Price"]), df["Present_Price"]
    return df, x, y

def pipeline() -> Pipeline:
    caracteristicas = ["Fuel_Type", "Selling_type", "Transmission"]
    caracteristicasNum = [ "Selling_Price", "Driven_kms", "Owner", "Age"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), caracteristicas),
            ('scaler', MinMaxScaler(), caracteristicasNum),
        ])
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression)), 
        ('regressor', LinearRegression() ), 
        ])
    return pipeline

def hiperParametros(pipeline, x, y):
    parametros = {
        "feature_selection__k": range(1, 12),
    }
    gridSearch = GridSearchCV(
        pipeline,
        parametros,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    return gridSearch.fit(x,y)

def guardar(model):
    os.makedirs('files/models', exist_ok=True)
    with gzip.open('files/models/model.pkl.gz', 'wb') as file:
        pickle.dump(model, file)

def metricas(pipeline, x_train, y_train, x_test, y_test):
    
    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)
    
    metricasTrain = {
        "type": "metrics",
        "dataset": "train",
        "r2": float(r2_score(y_train, y_train_pred)),
        "mse": float(mean_squared_error(y_train, y_train_pred)),
        "mad": float(median_absolute_error(y_train, y_train_pred)),
    }

    metricasTest = {
        "type": "metrics",
        "dataset": "test",
        "r2": float(r2_score(y_test, y_test_pred)),
        "mse": float(mean_squared_error(y_test, y_test_pred)),
        "mad": float(median_absolute_error(y_test, y_test_pred)),
    }

    return metricasTrain, metricasTest

def guardarMetricas(metrics_train, metrics_test, file_path="files/output/metrics.json"):
    metricas = [metrics_train, metrics_test]

    with open(file_path, "w") as f:
        for i in metricas:
            f.write(json.dumps(i) + "\n")


test = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
train = pd.read_csv("files/input/train_data.csv.zip", compression="zip")

test, x_test, y_test = limpiarDatos(test)
train, x_train, y_train = limpiarDatos(train)

modelo = pipeline()
modelo = hiperParametros(modelo, x_train, y_train)
guardar(modelo)

metrics_train, metrics_test = metricas(modelo, x_train, y_train, x_test, y_test)
guardarMetricas(metrics_train, metrics_test)