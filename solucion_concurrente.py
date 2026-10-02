import numpy as np
import threading
import time

def crear_matriz():
    rng = np.random.default_rng()
    matriz = rng.integers(low=1, high=100, size=(1000, 1000))
    return matriz

matriz_1000_x_1000 = crear_matriz()

resultados = [0] * 100

def sumar_bloque(filas: tuple[int, int], columnas: tuple[int, int], hilo_id):
    block_sum = 0

    for i in range(filas[0], filas[1]): 
        for j in range(columnas[0], columnas[1]):
            block_sum += matriz_1000_x_1000[i, j]

    resultados[hilo_id] = block_sum


def solucion_concurrente():

    hilos = []
    contador_hilos = 0

    tiempo_inicio = time.time()

    for row in range(0, 1000, 100):
        for cols_block in range(0, 1000, 100):

            # Cada bloque se le asigna a un hilo como dice el enunciado del taller
            hilos.append(threading.Thread(target=sumar_bloque, args=((row, row+100), (cols_block, cols_block +100), contador_hilos)))
            contador_hilos += 1
            print(f"Filas -> ({row}, {row+100}) \t Columnas ({cols_block}, {cols_block+100})")

    for hilo in hilos:
        hilo.start()

    # Esperar a que todos terminen
    for hilo in hilos:
        hilo.join()

    # Recolectar los resultados
    total = sum(resultados)

    tiempo_fin = time.time()

    # Utilizo la operación vectorizada de numpy para comprobar la suma
    total_esperado = np.sum(matriz_1000_x_1000)

    print(f"Resultado esperado calculado con numpy: {total_esperado}")
    print(f"Resultado calculado con hilos: {total}")
    print(f"Tiempo total {tiempo_fin-tiempo_inicio}")


def solucion_paralelo():
    pass


if __name__ == "__main__":
    solucion_concurrente()