import numpy as np
import time

dim = 10000

def crear_matriz():
    """
    Crea la matriz 1000x1000 con numeros del 1 al 100
    """
    rng = np.random.default_rng()
    matriz = rng.integers(low=1, high=100, size=(dim, dim))
    return matriz

matriz_d_10000 = crear_matriz()

def solucion_secuencial():
    tiempo_inicio = time.time()

    total = 0
    for i in range(dim):
        for j in range(dim):
            total += matriz_d_10000[i, j]

    tiempo_fin = time.time()

    # Utilizo la operación vectorizada de numpy para comprobar la suma
    total_esperado = np.sum(matriz_d_10000)

    print(f"Resultado esperado calculado con numpy: {total_esperado}")
    print(f"Resultado secuencial {total}")
    print(f"Tiempo secuencial {tiempo_fin-tiempo_inicio}")


if __name__ == "__main__":
    solucion_secuencial()