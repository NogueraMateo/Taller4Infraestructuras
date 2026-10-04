import time
import threading
from dataclasses import dataclass
from multiprocessing import cpu_count

import numpy as np


# Número de hilos utilizados para representar las unidades de ejecución
# disponibles en el sistema SMP.
TOTAL_HILOS = cpu_count()

# Dimensión de la matriz global.
DIM = 10000


@dataclass
class BloqueMatriz:
    """
    Describe lógicamente una subregión de la matriz.

    No crea una nueva matriz ni copia datos: únicamente almacena
    los límites de filas y columnas de cada bloque.
    """
    inicio_fil: int
    fin_fil: int
    inicio_col: int
    fin_col: int

    def __str__(self) -> str:
        return (
            f"Filas ({self.inicio_fil}, {self.fin_fil}) / "
            f"Cols ({self.inicio_col}, {self.fin_col})"
        )


def crear_matriz():
    """
    Crea la matriz de 10000x10000 que será compartida por todos los hilos.

    Como los hilos pertenecen al mismo proceso, todos tienen acceso al mismo
    espacio de memoria y, por tanto, pueden leer directamente esta matriz.
    """
    rng = np.random.default_rng()

    matriz = rng.integers(
        low=1,
        high=100,
        size=(DIM, DIM)
    )

    return matriz


# -------------------------------------------------------------------------
# MEMORIA COMPARTIDA ENTRE HILOS - COMPONENTE SMP
# -------------------------------------------------------------------------
# Todos los hilos acceden a esta misma matriz, ya que comparten el espacio
# de memoria del proceso.
matriz_d_10000 = crear_matriz()


# Cada hilo almacena su resultado parcial en una posición distinta.
# Al tener un único escritor por posición, no es necesario utilizar un Lock.
resultados = [0] * TOTAL_HILOS


def sumar_bloque(
    lista_bloques: list[BloqueMatriz],
    hilo_id: int
):
    """
    Procesa los bloques asignados a un hilo.

    Esta función combina dos niveles de procesamiento:

    SMP:
        Varios hilos ejecutan simultáneamente esta función sobre regiones
        distintas de la matriz compartida.

    SIMD:
        Dentro de cada hilo, np.sum procesa de forma vectorizada segmentos
        de las filas de la matriz. NumPy ejecuta el trabajo pesado en código
        compilado y puede aprovechar instrucciones SIMD soportadas por la CPU.
    """

    # Variable local del hilo.
    # No necesita sincronización porque ningún otro hilo accede a ella.
    block_sum = 0

    for bloque in lista_bloques:

        # Se recorren las filas correspondientes al bloque asignado.
        for fila in range(bloque.inicio_fil, bloque.fin_fil):

            # -----------------------------------------------------------------
            # COMPONENTE SIMD
            # -----------------------------------------------------------------
            # Se obtiene un vector de elementos consecutivos de una fila:
            #
            # matriz[fila, inicio_col:fin_col]
            #
            # np.sum ejecuta la reducción sobre ese vector mediante código
            # compilado. Dependiendo de la implementación de NumPy y del
            # hardware disponible, puede utilizar instrucciones vectoriales
            # SIMD para procesar múltiples elementos por instrucción.
            #
            # Además, NumPy puede liberar el GIL durante operaciones internas,
            # permitiendo que otros hilos ejecuten trabajo NumPy en paralelo.
            # -----------------------------------------------------------------
            suma_fila = np.sum(
                matriz_d_10000[
                    fila,
                    bloque.inicio_col:bloque.fin_col
                ]
            )

            block_sum += int(suma_fila)

    # Cada hilo escribe únicamente en su propia posición.
    resultados[hilo_id] = block_sum


def solucion_integradora():

    hilos: list[threading.Thread] = []

    tiempo_inicio = time.time()

    # -------------------------------------------------------------------------
    # PARTICIONAMIENTO DE LA MATRIZ
    # -------------------------------------------------------------------------
    # La matriz de 10000x10000 se divide lógicamente en bloques de 1000x1000.
    #
    # 10000 / 1000 = 10 bloques por dimensión
    # 10 x 10 = 100 bloques en total.
    #
    # Los bloques son descriptores de coordenadas; no se copian los datos.
    # -------------------------------------------------------------------------
    lista_bloques: list[BloqueMatriz] = []

    for row in range(0, DIM, 1000):
        for cols_block in range(0, DIM, 1000):

            bloque = BloqueMatriz(
                inicio_fil=row,
                fin_fil=row + 1000,
                inicio_col=cols_block,
                fin_col=cols_block + 1000
            )

            lista_bloques.append(bloque)

    # -------------------------------------------------------------------------
    # DISTRIBUCIÓN DEL TRABAJO - COMPONENTE SMP
    # -------------------------------------------------------------------------
    # Los 100 bloques se reparten entre los hilos disponibles.
    # Cada hilo procesará una lista parcial de bloques.
    # -------------------------------------------------------------------------
    bloques_por_hilo = len(lista_bloques) // TOTAL_HILOS
    resto = len(lista_bloques) % TOTAL_HILOS

    for hilo_id in range(TOTAL_HILOS):

        inicio_lista = hilo_id * bloques_por_hilo

        # Se mantiene la lógica original:
        # el último hilo recibe también los bloques restantes.
        fin_lista = (
            (hilo_id + 1) * bloques_por_hilo
            + (0 if hilo_id + 1 < TOTAL_HILOS else resto)
        )

        bloques_asignados = lista_bloques[inicio_lista:fin_lista]

        # ---------------------------------------------------------------------
        # COMPONENTE SMP
        # ---------------------------------------------------------------------
        # Cada hilo funciona como una unidad de ejecución que trabaja sobre
        # una parte independiente de la matriz compartida.
        # ---------------------------------------------------------------------
        hilo = threading.Thread(
            target=sumar_bloque,
            args=(bloques_asignados, hilo_id)
        )

        hilos.append(hilo)

    # Iniciar todos los hilos.
    # A partir de este punto pueden avanzar de forma concurrente y, durante
    # operaciones NumPy que liberen el GIL, puede existir paralelismo real.
    for hilo in hilos:
        hilo.start()

    # Barrera de sincronización:
    # el hilo principal espera hasta que todos los workers terminen.
    for hilo in hilos:
        hilo.join()

    # -------------------------------------------------------------------------
    # REDUCCIÓN FINAL
    # -------------------------------------------------------------------------
    # Cada hilo produjo una suma parcial.
    # El proceso principal combina esos resultados para obtener la suma total.
    # -------------------------------------------------------------------------
    total = sum(resultados)

    tiempo_fin = time.time()

    # Se utiliza NumPy únicamente para comprobar que el resultado obtenido
    # por la solución híbrida SMP-SIMD es correcto.
    total_esperado = np.sum(matriz_d_10000)

    print(
        f"Resultado esperado calculado con NumPy: {total_esperado}"
    )
    print(
        f"Resultado calculado con hilos: {total}"
    )
    print(
        f"Tiempo paralelo: {tiempo_fin - tiempo_inicio}"
    )


if __name__ == "__main__":
    solucion_integradora()