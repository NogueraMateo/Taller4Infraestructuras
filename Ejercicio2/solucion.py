import numpy as np
import time 

dim = 1000

def crear_matriz():
    """
    Crea la matriz 1000x1000 con numeros del 1 al 100
    """
    rng = np.random.default_rng()
    matriz = rng.integers(low=1, high=100, size=(dim, dim))
    return matriz


matriz1 = crear_matriz()
matriz2 = crear_matriz()

def multiplicar_matrices_numpy(A, B):
    return np.matmul(A, B)


def multiplicar_matrices(A, B):

    tiempo_inicio = time.time()
    C = np.zeros((dim, dim), dtype=np.int64)

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                C[i,j] += A[i,k] * B[k,j]

    tiempo_fin = time.time()

    print(f"Tiempo bucles {tiempo_fin-tiempo_inicio:.6f}")

    return C

if __name__ == "__main__":

    inicio_numpy = time.perf_counter()
    resultado_numpy = multiplicar_matrices_numpy(matriz1, matriz2)
    fin_numpy = time.perf_counter()

    inicio_bucles = time.perf_counter()
    resultado_bucles = multiplicar_matrices(matriz1, matriz2)
    fin_bucles = time.perf_counter()

    tiempo_numpy = fin_numpy - inicio_numpy
    tiempo_bucles = fin_bucles - inicio_bucles

    print(f"Tiempo NumPy: {tiempo_numpy:.6f} s")
    print(f"Tiempo bucles: {tiempo_bucles:.6f} s")

    print(
        "Resultados iguales:",
        np.array_equal(resultado_numpy, resultado_bucles)
    )
