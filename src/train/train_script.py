# Script que entrena modelos varios
# @Author: Benjamín Brito

import torch
import numpy as np
from CONTAC_transformers import Master


def train_with_params(
    name, exp, best_params, epochs, tolerancia=np.inf, paciencia=0, silent=False
):

    # buscamos en el experimento los prámetros del modelo y los emjores parámetros
    for item in exp:
        if item["name"] == "model_parameters":
            model_parameters = item["value"].split()
        if item["name"] == "train_parameters":
            train_parameters = item["value"].split()

    # Generamos un diccionario para los parámetros del modelo
    model_kwargs = dict()
    train_kwargs = dict()

    for item in exp:
        # if item["name"] == "name":
        #     name = item["value"]
        #     continue
        if item["name"] == "type":
            train_type = item["value"]
            continue
        if item["name"] == "datos":
            data = item["value"]
            continue

        for param in model_parameters:
            if item["name"] == param:
                if item["type"] == "fixed":
                    model_kwargs[param] = item["value"]
                else:
                    model_kwargs[param] = 0  # Placeholder para otros parámetros
                break
        for param in train_parameters:
            if item["name"] == param:
                if item["type"] == "fixed":
                    train_kwargs[param] = item["value"]
                else:
                    train_kwargs[param] = 0  # Placeholder para otros parámetros
                break

    # Incorporamos los parámetros encontreados
    for k, v in best_params.items():
        if k in model_kwargs.keys():
            model_kwargs[k] = v

        if k in train_kwargs.keys():
            train_kwargs[k] = v

    # Iniciamos el modelo y el entrenamiento
    m = Master()
    m.estructura(name, **model_kwargs)
    m.arq.to(train_kwargs["device"])

    # Actualizamos los prámetros de entrenamiento
    train_kwargs["loss"] = torch.nn.MSELoss()
    train_kwargs["optim"] = torch.optim.Adam(m.arq.parameters(), best_params["lr"])
    train_kwargs["epochs"] = epochs
    train_kwargs["tolerancia"] = tolerancia
    train_kwargs["paciencia"] = paciencia
    train_kwargs["silent"] = silent

    m.entrenar_modelo(data, best_params["batch"], train_type, **train_kwargs)

    return 0
