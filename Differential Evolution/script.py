# Declaración de librerías necesarias para graficar
# y hacer operaciones matemáticas
from mpl_toolkits import mplot3d
from matplotlib import cm
import random
import numpy as np
import matplotlib.pyplot as plt


# Función de prueba
def schwefel(x):
    """ Dominio de la función: -500 <= xi <= 500 (i = 1, ..., d).
        Mínimo global: f(x*) = 0, at x* = (420.9687, ..., 420.9687) """
    x = np.asarray_chkfinite(x)
    n = len(x)
    return(418.9829*n - sum(x * np.sin(np.sqrt(abs(x)))))


# Función para graficar la función de prueba y la evolución de sus mínimos
def plotFunction(arr):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    x = np.linspace(-500, 500, 80)
    y = np.linspace(-500, 500, 80)

    X, Y = np.meshgrid(x, y)
    Z = schwefel((X, Y))

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap=cm.jet, edgecolor='none', alpha=.25)

    for data in arr:
        # Si los parámetros contienen más de un elemento (puntos), significa
        # que se está haciendo un recorrido de las mejores partículas
        # por lo que se hace un coloreado distinto en cada situación
        xyz = (round(data[0], 4), round(data[1], 4))
        if(len(arr) > 1):
            def r(): return random.randint(0, 255)
            color = '#%02X%02X%02X' % (r(), r(), r())
            size = 20
            ax.set_title('Schwefel Sine - Evolution of global minima')
        else:
            color = 'r'
            size = 40
            ax.set_title('Schwefel Sine - Min Value')
        ax.scatter(xyz[0], xyz[1], round(schwefel(xyz), 4),
                   s=size, c=color, marker='.', zorder=10)


# Función para graficar la convergencia al mínimo global de la función
def plotConvergence(arr):
    x = list(list(zip(*arr))[0])
    y = list(list(zip(*arr))[0])
    z = []
    for data in arr:
        z.append(schwefel(data))

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(x, y, z, c='r', s=30)
    ax.plot(x, y, z, color='r')
    ax.set_title('Evolution of Algorithm')


# Función para delimitar los valores de la población
# a los límites reales de la función
def ensure_bounds(vec, bounds):
    vec_new = []
    # Por cada variable del vector:
    for i in range(len(vec)):
        # El valor excede el límite mínimo
        if(vec[i] < bounds[i][0]):
            vec_new.append(bounds[i][0])

        # El valor excede el límite máximo
        if(vec[i] > bounds[i][1]):
            vec_new.append(bounds[i][1])

        # El valor es el adecuado
        if(bounds[i][0] <= vec[i] <= bounds[i][1]):
            vec_new.append(vec[i])

    return(vec_new)


def diff_evolution(fitness, bounds, popsize, mutation, recomb, max_iter):
    """
        Parámetros
        ----------
        popsize : int
            Tamaño de la población, debe ser >= 4
        mutation : float
            Factor de mutación dentro del rango de [0,2]
        recomb : float
            Factor de recombinación dentro del rango de [0,1]
        max_iter : int
            Número máximo de generaciones
    """

    best_sol_array = []
    # Se genera una población de posibles soluciones (individuos)
    population = []
    for i in range(popsize):
        indv = []
        for j in range(len(bounds)):
            indv.append(random.uniform(bounds[j][0], bounds[j][1]))
        population.append(indv)

    # Por cada generación:
    for i in range(max_iter):
        gen_scores = []  # score keeping

        # Por cada individuo de la población
        for j in range(popsize):

            """
                Mutación
                ----------
                En este caso partícular, la mutación elegida para la implementación
                es la mutación dada por la fórmula v = x_1 + F(x_2 - x_3). Dado que,
                el nuevo vector donante producto de la mutación puede existir fuera
                de los límites especificados de la función, se emplea la función
                previamente definida para verificar y corregir esto. En caso de que
                se encuentre con una de estas situaciones, el vector donante se
                moverá al límite más cercano, ya sea el mínimo o el máximo.
            """
            # Se seleccionan tres posiciones aleatorias de vectores
            # sin incluir el vector actual (j)
            candidates = list(range(0, popsize))
            candidates.remove(j)
            random_index = random.sample(candidates, 3)
            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]

            # Individuo objetivo
            x_t = population[j]

            # Se sustrae x_3 de x_2
            # y se crea un nuevo vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

            # Se multiplica x_diff por el factor de mutación (F)
            # y se le añade a x_1
            v_donor = [x_1_i + mutation * x_diff_i for x_1_i,
                       x_diff_i in zip(x_1, x_diff)]
            v_donor = ensure_bounds(v_donor, bounds)

            """
                Recombinación

                Al recorrer cada posición del índice en el vector objetivo, se genera
                un valor aleatorio entre cero y uno. Si este valor aleatorio es menor
                que la tasa de recombinación, se produce una recombinación y cambiamos
                la variable actual en nuestro vector objetivo con la variable correspon-
                diente en el vector donante. Si el valor generado aleatoriamente es mayor
                que la tasa de recombinación, la recombinación no ocurre y la variable
                en el vector objetivo se deja sola. Este nuevo individuo descendiente
                se llama el vector de prueba.
            """
            v_trial = []
            for k in range(len(x_t)):
                crossover = random.random()
                if(crossover <= recomb):
                    v_trial.append(v_donor[k])

                else:
                    v_trial.append(x_t[k])

            """
                Seleccion voraz

                Consiste en evaluar nuestro nuevo vector de prueba (v_trial) contra
                individuo actualmente seleccionado (individuo objetivo, x_t), para
                ello se emplea usando una selección voraz.
            """

            score_trial = fitness(v_trial)
            score_target = fitness(x_t)

            if(score_trial < score_target):
                population[j] = v_trial
                gen_scores.append(score_trial)

            else:
                gen_scores.append(score_target)

        # Selección del mejor individuo de la población
        gen_sol = population[gen_scores.index(min(gen_scores))]
        best_sol_array.append(tuple(gen_sol))

    return gen_sol, best_sol_array


if __name__ == "__main__":
    # Límites conocidos de la función seno de schwefel
    bounds = [(-500, 500), (-500, 500)]

    global_best, sol_array = diff_evolution(
        schwefel, bounds, 15, 0.4, 0.8, 41)

    print("The best solution is: {}, {}".format(round(global_best[0], 4),
                                                round(global_best[1], 4)))

    plotConvergence(sol_array)
    plotFunction(sol_array)
    plotFunction([global_best])
    plt.show()
