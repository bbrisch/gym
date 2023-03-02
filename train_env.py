# Script que define el ambiente de entrenamiento
# @Author: Benjamín Brito
import argparse
import os
import src
import numpy as np

# Función que entrena los valores de la función
def main(
    experiment, model=None, epochs=None, tolerancia=None, paciencia=0, silent=False
):

    if not model:  # Caso en el que el modelo sea None o Lista vaciía
        model = [None] * len(experiment)
        epochs = [None] * len(experiment)

    else:
        assert len(experiment) == len(
            model
        ), "El largo de los experimentos y modelos no es el mismo"
        assert len(model) == len(
            epochs
        ), "El largo de los experimentos y épocas no es el mismo"

    # Iteramos entrenanando modelos
    for i, (exp, mod, ep) in enumerate(zip(experiment, model, epochs)):
        print("\n")
        print(f"ITER {i} | EXPERIMENTO {exp} | MODELO {mod} | EPOCHS {ep}")
        if not os.path.exists(
            f"resultados/experimemtos/{exp}"
        ):  # Caso en el que no existe la experiencia
            parameters = np.load(f"resultados/templates/{exp}.npy", allow_pickle=True)
            src.tune_model(parameters)

        if not mod is None:
            parameters = np.load(f"resultados/templates/{exp}.npy", allow_pickle=True)
            best_params = np.load(
                f"resultados/experimentos/{exp}.npy", allow_pickle=True
            )

            src.train_with_params(
                parameters, best_params, ep, tolerancia, paciencia, silent
            )


if __name__ == "__main__":

    argParser = argparse.ArgumentParser()

    # Se añaden argumentos al parser
    argParser.add_argument("-e", required=True)
    argParser.add_argument("-m", required=False)
    argParser.add_argument("-ep", required=False)
    argParser.add_argument("-t", required=False)
    argParser.add_argument("-p", required=False)
    argParser.add_argument("-s", required=False)

    args = argParser.parse_args()
    args = vars(args)
    # print(args)
    for k, v in args.items():
        if not v is None:
            if k in ["m", "e"]:
                args[k] = v.split(",")
            elif k in ["ep"]:
                args[k] = [int(e) for e in v.split(",")]
            else:
                args[k] = int(v)

    # Ajustamos las llaves con los nomnbres corresponidentes
    trad = {
        "e": "experiment",
        "m": "model",
        "ep": "epochs",
        "t": "tolerancia",
        "p": "paciencia",
        "s": "silent",
    }
    for k, v in trad.items():
        if args[k] is None:
            _ = args.pop(k)
        else:
            args[v] = args.pop(k)

    # Generamos la estructura de carpetas y esperamos a que el usuario confirme el setup
    if not os.path.exists("resultados"):
        print("Iniciando setup...")
        os.makedirs("resultados/templates")
        os.makedirs("resultados/experimentos")
        os.makedirs("resultados/modelos")

        # Capturamos el programa hasta que el usuario esté listo
        text = """\nGenere los templates en la carpeta correspondiente.\nPresione cualquier tecla para continuar."""
        input(text)

    print(args)
    main(**args)
