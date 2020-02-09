# Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import os

from numpy.random import uniform
from pprint import pprint
from mpl_toolkits import mplot3d
from matplotlib import cm


# Función objetivo
def target_function():
    return


# Inicialización de las variables y del objetivo
def initial_sources(food_sources=3, min_values=[-500, -500], max_values=[500, 500], target_function=target_function):
    sources = np.zeros((food_sources, len(min_values) + 1))
    for i in range(0, food_sources):
        for j in range(0, len(min_values)):
            sources[i, j] = random.uniform(min_values[j], max_values[j])
        sources[i, -1] = target_function(sources[i, 0:sources.shape[1]-1])
    return sources


# Función para calcular el valor más óptimo
def fitness_calc(function_value):
    if(function_value >= 0):
        fitness_value = 1.0/(1.0 + function_value)
    else:
        fitness_value = 1.0 + abs(function_value)
    return fitness_value


# Función para buscar en el espacio de búsqueda la función óptima
def fitness_function(searching_in_sources):
    fitness = np.zeros((searching_in_sources.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        #fitness[i,0] = 1/(1+ searching_in_sources[i,-1] + abs(searching_in_sources[:,-1].min()))
        fitness[i, 0] = fitness_calc(searching_in_sources[i, -1])
    fit_sum = fitness[:, 0].sum()
    fitness[0, 1] = fitness[0, 0]
    for i in range(1, fitness.shape[0]):
        fitness[i, 1] = (fitness[i, 0] + fitness[i-1, 1])
    for i in range(0, fitness.shape[0]):
        fitness[i, 1] = fitness[i, 1]/fit_sum
    return fitness


# Selección de ruleta
def roulette_wheel(fitness):
    ix = 0
    random = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
            ix = i
            break
    return ix


# Abeja trabajadora
def employed_bee(sources, min_values=[-500, -500], max_values=[500, 500], target_function=target_function):
    searching_in_sources = np.copy(sources)
    new_solution = np.zeros((1, len(min_values)))
    trial = np.zeros((sources.shape[0], 1))
    for i in range(0, searching_in_sources.shape[0]):
        phi = random.uniform(-1, 1)
        j = np.random.randint(len(min_values), size=1)[0]
        k = np.random.randint(searching_in_sources.shape[0], size=1)[0]
        while i == k:
            k = np.random.randint(searching_in_sources.shape[0], size=1)[0]
        xij = searching_in_sources[i, j]
        xkj = searching_in_sources[k, j]
        vij = xij + phi*(xij - xkj)
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = searching_in_sources[i, variable]
        new_solution[0, j] = np.clip(vij, min_values[j], max_values[j])
        new_function_value = target_function(
            new_solution[0, 0:new_solution.shape[1]])
        if (fitness_calc(new_function_value) > fitness_calc(searching_in_sources[i, -1])):
            searching_in_sources[i, j] = new_solution[0, j]
            searching_in_sources[i, -1] = new_function_value
        else:
            trial[i, 0] = trial[i, 0] + 1
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = 0.0
    return searching_in_sources, trial


# Abeja en espera
def outlooker_bee(searching_in_sources, fitness, trial, min_values=[-500, -500], max_values=[500, 500], target_function=target_function):
    improving_sources = np.copy(searching_in_sources)
    new_solution = np.zeros((1, len(min_values)))
    trial_update = np.copy(trial)
    for repeat in range(0, improving_sources.shape[0]):
        i = roulette_wheel(fitness)
        phi = random.uniform(-1, 1)
        j = np.random.randint(len(min_values), size=1)[0]
        k = np.random.randint(improving_sources.shape[0], size=1)[0]
        while i == k:
            k = np.random.randint(improving_sources.shape[0], size=1)[0]
        xij = improving_sources[i, j]
        xkj = improving_sources[k, j]
        vij = xij + phi*(xij - xkj)
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = improving_sources[i, variable]
        new_solution[0, j] = np.clip(vij,  min_values[j], max_values[j])
        new_function_value = target_function(
            new_solution[0, 0:new_solution.shape[1]])
        if (fitness_calc(new_function_value) > fitness_calc(improving_sources[i, -1])):
            improving_sources[i, j] = new_solution[0, j]
            improving_sources[i, -1] = new_function_value
            trial_update[i, 0] = 0
        else:
            trial_update[i, 0] = trial_update[i, 0] + 1
        for variable in range(0, len(min_values)):
            new_solution[0, variable] = 0.0
    return improving_sources, trial_update


# Abeja exploradora
def scouter_bee(improving_sources, trial_update, limit=3, target_function=target_function):
    for i in range(0, improving_sources.shape[0]):
        if (trial_update[i, 0] > limit):
            for j in range(0, improving_sources.shape[1] - 1):
                improving_sources[i, j] = np.random.normal(0, 1, 1)[0]
            function_value = target_function(
                improving_sources[i, 0:improving_sources.shape[1]-1])
            improving_sources[i, -1] = function_value
    return improving_sources


# Colonia de abejas
def artificial_bee_colony_optimization(food_sources=3, iterations=50, min_values=[-500, -500], max_values=[500, 500], employed_bees=3, outlookers_bees=3, limit=3, target_function=target_function):
    count = 0
    best_value = float("inf")
    evol = []
    sources = initial_sources(food_sources=food_sources, min_values=min_values,
                              max_values=max_values, target_function=target_function)
    fitness = fitness_function(sources)
    while (count <= iterations):
        if (count > 0):
            evol.append(tuple([value[1], value[1]]))
        e_bee = employed_bee(sources, min_values=min_values,
                             max_values=max_values, target_function=target_function)
        for i in range(0, employed_bees - 1):
            e_bee = employed_bee(e_bee[0], min_values=min_values,
                                 max_values=max_values, target_function=target_function)
        fitness = fitness_function(e_bee[0])
        o_bee = outlooker_bee(e_bee[0], fitness, e_bee[1], min_values=min_values,
                              max_values=max_values, target_function=target_function)
        for i in range(0, outlookers_bees - 1):
            o_bee = outlooker_bee(o_bee[0], fitness, o_bee[1], min_values=min_values,
                                  max_values=max_values, target_function=target_function)
        value = np.copy(o_bee[0][o_bee[0][:, -1].argsort()][0, :])
        if (best_value > value[-1]):
            best_solution = np.copy(value)
            best_value = np.copy(value[-1])
        sources = scouter_bee(
            o_bee[0], o_bee[1], limit=limit, target_function=target_function)
        fitness = fitness_function(sources)
        count = count + 1
    return(best_solution, evol)


def schwefel(variables_values=[0, 0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + \
            ((variables_values[i] * np.sin(np.sqrt(abs(variables_values[i])))))
    return(418.9829*2 - func_value)


# Función de prueba
def schwefel_plot(x):
    """ Dominio de la función: -500 <= xi <= 500 (i = 1, ..., d).
        Mínimo global: f(x*) = 0, at x* = (420.9687, ..., 420.9687) """
    x = np.asarray_chkfinite(x)
    n = len(x)
    return(418.9829*n - sum(x * np.sin(np.sqrt(abs(x)))))


# Función para graficar la función de prueba y la evolución de sus mínimos
def plotFunction(arr, num):
    fig = plt.figure(num)
    mark_arr = ['>', '<', '^', 'v']
    ax = plt.axes(projection="3d")
    x = np.linspace(-500, 500, 80)
    y = np.linspace(-500, 500, 80)

    X, Y = np.meshgrid(x, y)
    Z = schwefel_plot((X, Y))

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=cm.jet, edgecolor='none', alpha=.25)

    i = 0
    for data in arr:
        xyz = (round(data[0], 4), round(data[1], 4))
        # Si los parámetros contienen más de un elemento (puntos), significa
        # que se está haciendo un recorrido de las mejores partículas
        # por lo que se hace un coloreado distinto en cada situación
        if(len(arr) > 1):
            def r(): return random.randint(0, 255)
            color = '#%02X%02X%02X' % (r(), r(), r())
            marker = mark_arr[i]
            size = 10
            ax.set_title('Schwefel Sine - Evolution of global minima')
        else:
            color = 'r'
            marker = '.'
            size = 40
            ax.set_title('Schwefel Sine - Min Value')
        ax.scatter(xyz[0], xyz[1], round(schwefel_plot(xyz),
                                         4), s=size, c=color, marker=marker, zorder=10)
        i += 1
        if(i > 3):
            i = 0


# Función para graficar la convergencia al mínimo global de la función
def plotConvergence(arr):
    x = []
    y = []
    i = 1
    for data in arr:
        x.append(i)
        y.append(schwefel_plot(data))
        i += 1

    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)

    xs = x
    ys = y

    ax.plot(xs, ys, marker='.')
    ax.set_title("Evolution of the algorithm")
    ax.set_ylabel("Function value")
    ax.set_xlabel("Iterations")


if __name__ == "__main__":
    best, evol = artificial_bee_colony_optimization(food_sources=300, iterations=10, min_values=[-500, -500], max_values=[
        500, 500], employed_bees=50, outlookers_bees=50, limit=40, target_function=schwefel)

    print("The best solution is: [{}, {}] with a value of {}".format(
        best[1], best[1], schwefel_plot([best[1], best[1]])))

    plotConvergence(evol)
    plotFunction(evol, 2)
    plotFunction([[best[1], best[1]]], 3)

    plt.show()
