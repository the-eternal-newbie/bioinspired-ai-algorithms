# Declaración de librerías necesarias para graficar
# y hacer operaciones matemáticas
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from numpy.random import uniform
from pprint import pprint
from mpl_toolkits import mplot3d
from matplotlib import cm


class clonalg():
    # Función objetivo o de afinidad (fitness)
    def schwefel(p_i):
        """
            Dominio de la función: -500 <= xi <= 500 (i = 1, ..., d).
            Mínimo global: f(x*) = 0, at x* = (420.9687, ..., 420.9687)
        """
        xy = np.asarray_chkfinite(p_i)
        n = len(xy)
        return(418.9829*n - sum(xy * np.sin(np.sqrt(abs(xy)))))

    # Función para la creación de anticuerpos (individuos) aleatorios
    def create_random_cells(population_size, problem_size, b_lo, b_up):
        population = [uniform(low=b_lo, high=b_up, size=problem_size)
                      for x in range(population_size)]
        return(population)

    # Función de clonación de anticuerpos
    def clone(p_i, clone_rate):
        clone_num = int(clone_rate / p_i[1])
        clones = [(p_i[0], p_i[1]) for x in range(clone_num)]
        return(clones)

    # Función de hipermutación
    def hypermutate(p_i, mutation_rate, b_lo, b_up):
        if(uniform() <= p_i[1] / (mutation_rate * 100)):
            ind_tmp = []
            for gen in p_i[0]:
                if(uniform() <= p_i[1] / (mutation_rate * 100)):
                    ind_tmp.append(uniform(low=b_lo, high=b_up))
                else:
                    ind_tmp.append(gen)
            return(np.array(ind_tmp), clonalg.schwefel(ind_tmp))
        else:
            return(p_i)

    # Función de selección de anticuerpos
    def select(pop, pop_clones, pop_size):
        population = pop + pop_clones
        population = sorted(population, key=lambda x: x[1])[:pop_size]
        return(population)

    # Función para reemplazar los anticuerpos poco afines
    def replace(population, population_rand, population_size):
        population = population + population_rand
        population = sorted(population, key=lambda x: x[1])[:population_size]
        return(population)


# Función para graficar la función de prueba y la evolución de sus mínimos
def plotFunction(arr, num):
    fig = plt.figure(num)
    ax = plt.axes(projection="3d")
    x = np.linspace(-500, 500, 80)
    y = np.linspace(-500, 500, 80)

    X, Y = np.meshgrid(x, y)
    Z = clonalg.schwefel((X, Y))

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=cm.jet, edgecolor='none', alpha=.25)

    for data in arr:
        xyz = (round(data[0], 4), round(data[1], 4))
        # Si los parámetros contienen más de un elemento (puntos), significa
        # que se está haciendo un recorrido de las mejores partículas
        # por lo que se hace un coloreado distinto en cada situación
        if(len(arr) > 1):
            def r(): return random.randint(0, 255)
            color = '#%02X%02X%02X' % (r(), r(), r())
            size = 20
            ax.set_title('Schwefel Sine - Evolution of global minima')
        else:
            color = 'r'
            size = 40
            ax.set_title('Schwefel Sine - Min Value')
        ax.scatter(xyz[0], xyz[1], round(clonalg.schwefel(xyz),
                                         4), s=size, c=color, marker='.', zorder=10)


if __name__ == "__main__":
    """
        Parámetros del algoritmo
        ------------------------
        b_lo, b_up : int[matrix]
            Dominio de la función
        population_size : int
            Tamaño de la población de células (anticuerpos)
        selection_size: int
            Tamaño de selección de los anticuerpos
        random_cells_num: int
            Número de células aleatorias de la hipermutación
        problem_size: int
            Número de dimensiones del problema
        clone_rate: int
            Tasa de clonación
        mutation_rate: float
            Tasa de mutación
        max_iter:
            Máximo de iteraciones del ciclo
    """
    b_lo, b_up = (-500, 500)

    population_size = 50
    selection_size = 12
    problem_size = 2
    random_cells_num = 85
    clone_rate = 45
    mutation_rate = 0.05
    max_iter = 1500
    stop = 0

    # Población <- Genera anticuerpos de manera aleatoria a partir
    # del tamaño de la población y de las dimensiones del problema
    population = clonalg.create_random_cells(
        population_size, problem_size, b_lo, b_up)
    best_affinity_it = []

    img_num = 1
    # Mientras la condición de paro no se cumpla:
    while stop != max_iter:
        # Se calcula la afinidad de p_i (fitness/función objetivo)
        population_affinity = [(p_i, clonalg.schwefel(p_i))
                               for p_i in population]
        # Se ordena la afinidad de la población
        populatin_affinity = sorted(population_affinity, key=lambda x: x[1])

        # Para efectos de comparación, se añaden los cinco mejores
        # valores de la población de cada iteración a un arreglo
        # de mejores valores (valores óptimos obtenidos)
        best_affinity_it.append(populatin_affinity[:5])

        # Selección de los anticuerpos <- Selecciona una cantidad
        # fija de anticuerpos (selection_size)
        population_select = populatin_affinity[:selection_size]

        # Se clonan los anticuerpos de manera proporcional
        # a la afinidad y a partir de la tasa de clonación
        population_clones = []
        for p_i in population_select:
            p_i_clones = clonalg.clone(p_i, clone_rate)
            population_clones += p_i_clones

        """
            Hipermutación
            -------------
            El conjunto clonal resultante compite con la población
            de anticuerpos existente para pertenecer a la próxima
            generación, de esta manera, los miembros de la población
            con baja afinidad se reemplazan por anticuerpos generados
            aleatoriamente.
        """
        pop_clones_tmp = []
        for p_i in population_clones:
            ind_tmp = clonalg.hypermutate(p_i, mutation_rate, b_lo, b_up)
            pop_clones_tmp.append(ind_tmp)
        population_clones = pop_clones_tmp
        del pop_clones_tmp

        # Se selecciona a la población con baja afinidad
        population = clonalg.select(
            populatin_affinity, population_clones, population_size)
        # Se genera una población aleatoria
        population_rand = clonalg.create_random_cells(
            random_cells_num, problem_size, b_lo, b_up)
        population_rand_affinity = [(p_i, clonalg.schwefel(p_i))
                                    for p_i in population_rand]
        population_rand_affinity = sorted(
            population_rand_affinity, key=lambda x: x[1])
        # Se remplaza la anterior población de anticuerpos
        population = clonalg.replace(
            population_affinity, population_rand_affinity, population_size)
        population = [p_i[0] for p_i in population]

        stop += 1

        # Pequeña condicional para ver el progreso del algoritmo en el plano
        if(stop % 50 == 0):
            plotFunction(population, 2)
            plt.pause(0.05)
            plt.savefig('evol_{}.png'.format(img_num))
            img_num += 1

    plt.show()

    # Del anterior ciclo, se obtiene el promedio de los mejores
    # cinco individuos retornados por cada iteración
    bests_mean = []
    iterations = [i for i in range(1500)]
    for pop_it in best_affinity_it:
        bests_mean.append(np.mean([p_i[1] for p_i in pop_it]))

    # Se grafica la evolución de la optimización de los valores
    plt.title("Evolution of Optimization", fontsize=12)
    plt.plot(iterations, bests_mean)
    plotFunction([population_affinity[0][0]], 3)

    print("The best solution is: [{}, {}] with a value of {}".format(round(population_affinity[0][0][0], 4),
                                                                     round(population_affinity[0][0][1], 4), population_affinity[0][1]))
    plt.show()
