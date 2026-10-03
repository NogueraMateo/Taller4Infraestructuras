from dataclasses import dataclass
from multiprocessing.sharedctypes import Synchronized
from typing import Any

import numpy as np
from multiprocessing import cpu_count, shared_memory, Value, Lock, Process, current_process
import time


@dataclass
class BloqueMatriz:
    inicio_fil: int
    fin_fil: int
    inicio_col: int
    fin_col: int

    def __str__(self) -> str:
            return f"Filas ({self.inicio_fil}, {self.fin_fil}) / Cols ({self.inicio_col}, {self.fin_col})"
    
dim = 1000 # Dimensión de la matriz

    
def sumar_lista_bloques(
    lista_parcial: list[BloqueMatriz],
    shared_matrix_memory_name: str, 
    shared_counter: Synchronized[Any]
):
    existing_matrix_shared_memory = shared_memory.SharedMemory(
        name=shared_matrix_memory_name
        )

    shared_matrix = np.ndarray(
        (dim, dim), 
        dtype=np.int64, 
        buffer=existing_matrix_shared_memory.buf
        )
    
    block_sum = 0
    for b in lista_parcial:
        for i in range(b.inicio_fil, b.fin_fil): 
            for j in range(b.inicio_col, b.fin_col):
                block_sum += shared_matrix[i, j]

    with shared_counter.get_lock():
        shared_counter.value += block_sum

    existing_matrix_shared_memory.close()


def obtener_bloques() -> list[BloqueMatriz]:
    """
    Devuelve una lista con 100 objetos de datos. Cada objeto tiene
    los datos que describe un bloque de la matriz.
    """
    lista_bloques = []
    for row in range(0, dim, 100):
        for cols_block in range(0, dim, 100):
            lista_bloques.append(BloqueMatriz(row, row+100, cols_block, cols_block + 100))

    return lista_bloques


def crear_matriz_compartida():
    rng = np.random.default_rng()
    matriz = rng.integers(low=1, high=100, size=(dim, dim))

    shm = shared_memory.SharedMemory(create=True, size=matriz.nbytes)

    np_array = np.ndarray(matriz.shape,dtype=np.int64, buffer=shm.buf)
    np_array[:] = matriz[:]

    return shm, np_array

if __name__ == "__main__":
    if current_process().name == "MainProcess":
        
        shr, matriz = crear_matriz_compartida()

        resultado_suma = Value("q", 0)

        bloques = obtener_bloques()

        bloques_por_proceso = len(bloques) // cpu_count()

        resto = len(bloques) % cpu_count()

        procesos: list[Process] = []

        tiempo_inicio = time.time()
        
        for i in range(cpu_count()):
            inicio = i*bloques_por_proceso
            fin = (i+1)*(bloques_por_proceso) if i+1 < cpu_count() else (i+1)*(bloques_por_proceso) + resto
            lista_parcial = bloques[inicio:fin]

            process = Process(target=sumar_lista_bloques, args=(lista_parcial, shr.name, resultado_suma))

            procesos.append(process)
            process.start()

        for process in procesos:
            process.join()


        tiempo_fin = time.time()
        

        # Utilizo la operación vectorizada de numpy para comprobar la suma
        total_esperado = np.sum(matriz)

        print(f"Resultado esperado calculado con numpy: {total_esperado}")
        print(f"Resultado calculado paralelamente: {resultado_suma.value}")
        print(f"Tiempo paralelo {tiempo_fin-tiempo_inicio}")

        shr.close()
        shr.unlink()

