# Script con funciones para optimización bayesiana
# @Author: Benjamín Brito

import torch
import torch.nn as nn

from tqdm import tqdm
from CONTAC_transformers import Master, Datos_RAM


# Función de entrenameinto estádar para transformers
def net_train(model, train_loader, parameters, silent=True, loss=nn.MSELoss()):

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

        #   # Enmascaramiento de la entrada
        #   for _ in range(parameters.get('realizaciones', 0)):
        #       # Generación de la máscara
        #       mascara = torch.bernoulli(parameters.get('probabilidad', 2/3)*torch.ones(ejemplo.shape)).float().to(device)
        #       ejemplo_m = ejemplo*mascara

        #       # Cálculo de la pérdida
        #       output = model(ejemplo_m)
        #       perdida = loss(output,respuesta)
        #       cumulative_loss.append(perdida.item())

        #       # Backprop
        #       optim.zero_grad()
        #       perdida.backward()
        #       optim.step()

    return model


# Función que evalúa el modelo y retorna el promedio de la función de pérdida en los batches
def net_eval(model, val_loader, parameters, silent=True, loss=nn.MSELoss()):

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
    # Generamos un modelo
    m = (
        Master()
    )  # Se genera un modelo con la estructura predeterminada y prámetros por tunear

    m.estructura(
        parameterization.get("name"),
        **{
            "model": "pred",
            "seq_leng": 5,
            "target_features": 1,
            "target_leng": 1,
            "in_features": 23,
            "pdrop": 0.2,
            "model_features": parameterization.get("model_features"),
            "h_features": parameterization.get("h_features"),
            "n_heads": parameterization.get("n_heads", 1),
            "n_layers": parameterization.get("n_layers"),
        },
    )
    return m.arq  # Retorna la red sin entrenar


# Función con el ciclo de entrenamiento y evaluación de los modelos
def train_evaluate(parameterization):
    # Abrimos los datos
    print(parameterization.get("datos"))
    dataset_train = Datos_RAM(parameterization.get("datos") + "/train.npy")
    dataset_val = Datos_RAM(parameterization.get("datos") + "/val.npy")

    # Generamos los dataloaders
    dataload_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=parameterization.get("batch", 512), shuffle=True
    )
    dataload_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=parameterization.get("batch", 512), shuffle=False
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
