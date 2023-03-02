# Script con funciones para optimización bayesiana
# @Author: Benjamín Brito

import os
import torch

import torch.nn as nn
import numpy as np

from tqdm import tqdm
from CONTAC_transformers import Master, Datos_RAM
from ax.service.managed_loop import optimize


# Función de entrenameinto estádar para transformers
def net_train(model, train_loader, parameters, silent=False, loss=nn.MSELoss()):

    optim = torch.optim.Adam(model.parameters(), lr=parameters.get("lr", 0.001))
    device = parameters.get("device", "cpu")
    model.to(device)
    for epoch in range(int(parameters.get("epochs", 5))):

        # Iteraciones con datos de train
        cumulative_loss = []
        model.train()
        for (ejemplo, respuesta) in tqdm(
            train_loader, desc=f"Entrenamiento época {epoch}", disable=silent
        ):
            ejemplo = ejemplo.to(device)
            respuesta = respuesta.to(device)

            # Cálculo de la pérdida
            output = model(ejemplo)
            perdida = loss(output, respuesta)
            cumulative_loss.append(perdida.item())

            # Backprop
            optim.zero_grad()
            perdida.backward()
            optim.step()

    return model


# Función que evalúa el modelo y retorna el promedio de la función de pérdida en los batches
def net_eval(model, val_loader, parameters, silent=False, loss=nn.MSELoss()):

    device = parameters.get("device", "cpu")
    model.to(device)
    for epoch in range(parameters.get("epochs", 5)):

        # Iteraciones con datos de train
        cumulative_loss = []
        model.eval()
        for (ejemplo, respuesta) in tqdm(
            val_loader, desc=f"Validación época {epoch}", disable=silent
        ):
            ejemplo = ejemplo.to(device)
            respuesta = respuesta.to(device)

            # Cálculo de la pérdida
            output = model(ejemplo)
            perdida = loss(output, respuesta)
            cumulative_loss.append(perdida.item())

    return sum(cumulative_loss) / len(cumulative_loss)


# Función que inicializa y entrega una red sin entrenar
def init_net(parameterization):

    model_parameters = {
        param: parameterization.get(param)
        for param in parameterization.get("model_parameters").split()
    }

    # Generamos un modelo
    m = (
        Master()
    )  # Se genera un modelo con la estructura predeterminada y prámetros por tunear

    m.estructura(parameterization.get("name"), **model_parameters)
    return m.arq  # Retorna la red sin entrenar


# Función con el ciclo de entrenamiento y evaluación de los modelos
def train_evaluate(parameterization):
    # Abrimos los datos
    print(parameterization.get("datos"))
    dataset_train = Datos_RAM(parameterization.get("datos") + "/train.npy")
    dataset_val = Datos_RAM(parameterization.get("datos") + "/val.npy")

    # Generamos los dataloaders
    dataload_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=parameterization.get("batch", 512),
        shuffle=True,
        num_workers=2,
    )
    dataload_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=parameterization.get("batch", 512),
        shuffle=False,
        num_workers=2,
    )

    # Get neural net
    untrained_net = init_net(parameterization)

    # train
    trained_net = net_train(
        model=untrained_net, train_loader=dataload_train, parameters=parameterization
    )

    # return the accuracy of the model as it was trained in this run
    return net_eval(
        model=trained_net, val_loader=dataload_val, parameters=parameterization
    )


# Función que maneja el pipeline y optimiza. Esta va a ser llamada por el bot
def tune_model(parameterization, trials):

    best_parameters, values, experiment, model = optimize(
        parameters=parameterization,
        evaluation_function=train_evaluate,
        objective_name="validation_loss",
        minimize=True,
        total_trials=trials,
    )
    for d in parameterization:
        if d["name"] == "id":
            id = d["value"]
            break

    # print('path:',path)
    if not os.path.exists("resultados\experimentos/" + id):
        os.makedirs("resultados\experimentos/" + id)

    torch.save(values, "resultados\experimentos/" + id + "/values.pt")
    torch.save(experiment, "resultados\experimentos/" + id + "/experiment.pt")
    torch.save(model, "resultados\experimentos/" + id + "/model.pt")
    torch.save(best_parameters, "resultados\experimentos/" + id + "/h_params.pt")


if __name__ == "__main__":
    print("Gargando datos...")
    path_data = "C:/Users/CAEX/Documents/Archivos_Martín/Anglo_CF/gym-dev/datos"
    id = 0

    parameters = [
        # parámetros del experimento
        {"name": "id", "type": "fixed", "value": f"{id}", "value_type": "str"},
        {
            "name": "datos",
            "type": "fixed",
            "value": f"{path_data}",
            "value_type": "str",
        },
        # Parámetros fijos del entrenameinto
        {
            "name": "train_parameters",  # Lista con parámetros de entrenamiento
            "type": "fixed",
            "value": "device",
            "value_type": "str",
        },
        {
            "name": "type",
            "type": "fixed",
            "value": "std",
            "value_type": "str",
        },  # Tipo de entrenamiento
        {"name": "device", "type": "fixed", "value": "cuda", "value_type": "str"},
        {"name": "epochs", "type": "fixed", "value": 5, "value_type": "int"},
        # Parámetros adaptables del entrenamiento
        {
            "name": "lr",
            "type": "range",
            "bounds": [1e-6, 1e-3],
            "log_scale": True,
            "value_type": "float",
        },
        {
            "name": "batch",
            "type": "choice",
            "values": [2**7, 2**8, 2**9, 2**10],
            "value_type": "int",
            "is_ordered": True,
        },
        # Parámetros del modelo
        {
            "name": "model_parameters",  # Lista con parámetros del modelo
            "type": "fixed",
            "value": "model seq_leng target_features target_leng in_features pdrop model_features h_features n_heads n_layers",
            "value_type": "str",
        },
        {
            "name": "name",  # Nombre del modelo
            "type": "fixed",
            "value": f"resultados/experimentos/{id}",
            "value_type": "str",
        },
        # parámetros fijos (dependen del problema y arquitectura a usar)
        {
            "name": "model",
            "type": "fixed",
            "value": "pred",
            "value_type": "str",
        },
        {
            "name": "seq_leng",
            "type": "fixed",
            "value": 60,
            "value_type": "int",
        },
        {
            "name": "target_features",
            "type": "fixed",
            "value": 2,
            "value_type": "int",
        },
        {
            "name": "target_leng",
            "type": "fixed",
            "value": 30,
            "value_type": "int",
        },
        {
            "name": "in_features",
            "type": "fixed",
            "value": 20,
            "value_type": "int",
        },
        {
            "name": "pdrop",
            "type": "fixed",
            "value": 0.2,
            "value_type": "int",
        },
        # Parámetros del modelo a optimizar
        {
            "name": "model_features",
            "type": "range",
            "bounds": [40, 60],
            "value_type": "int",
        },
        {
            "name": "h_features",
            "type": "range",
            "bounds": [60, 80],
            "value_type": "int",
        },
        {"name": "n_layers", "type": "range", "bounds": [4, 8], "value_type": "int"},
        {"name": "n_heads", "type": "range", "bounds": [1, 5], "value_type": "int"},
    ]

    path_save = (
        "C:/Users/CAEX/Documents/Archivos_Martín/Anglo_CF/gym-dev/resultados/templates"
    )
    print("Guardando datos...")
    np.save(path_save + f"/{id}.npy", parameters, allow_pickle=True)
    print("¡Datos guardados!")
