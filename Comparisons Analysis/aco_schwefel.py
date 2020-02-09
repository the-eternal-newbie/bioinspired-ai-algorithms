# importar librerías para el programa
from math import pi
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


# función de Prueba
def schwefel(x):
    """ Dominio de la función: -500 <= xi <= 500 (i = 1, ..., d).
        Mínimo global: f(x*) = 0, at x* = (420.9687, ..., 420.9687) """
    x = np.asarray_chkfinite(x)
    n = len(x)
    return(418.9829*n - sum(x * np.sin(np.sqrt(abs(x)))))


class ant_colony:
    """ Class containing the Ant Colony Optimization for Continuous Domains """

    # constructor de la clase
    def __init__(self):
        """ parámetros iniciales del algoritmo """

        # número máximo de iteraciones
        self.max_iter = 100
        # tamaño de la población
        self.pop_size = 5
        # tamaño del registro (de las soluciones)
        self.k = 50
        # localidad de búsqueda
        self.q = 0.1
        # velocidad de convergencia
        self.xi = 0.85

        """ definición initial del problema (nulo) """

        # número de variables (dimensiones)
        self.num_var = 2
        # límites del dominio
        self.var_ranges = [[0, 1], [0, 1]]
        # función de costo (fitness)
        self.cost_function = None

        """ resultados de la optimización """
        # registro de las soluciones
        self.SA = None
        # mejor solución del registro
        self.best_solution = None
        # arreglo para almacenar los valores mínimos
        self.minima_array = [[], []]
        # arreglo correlacionado con los mínimos para saber su iteración
        self.iteration_array = []

    # asignación de las variables de dimensión y dominio
    def set_variables(self, nvar, ranges):
        if len(ranges) != nvar:
            print("Error, number of variables and ranges does not match")
        else:
            self.num_var = nvar
            self.var_ranges = ranges
            self.SA = np.zeros((self.k, self.num_var + 1))

    # asignación de la función de costo a evaluar
    def set_cost(self, costf):
        self.cost_function = costf

    # asignación de los parámetros del algoritmo (máximo de iteraciones, tamaño de población,
    # tamaño del registro, localidad y velocidad)
    def set_parameters(self, max_iter, pop_size, k, q, xi):
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.k = k
        self.q = q
        self.xi = xi

    # función de ruleta de selección
    def _biased_selection(self, probabilities):
        # retorna un índice basado en un set de probabilidades
        r = np.random.uniform(0, sum(probabilities))
        for i, f in enumerate(probabilities):
            r -= f
            if r <= 0:
                return i

    # función de optimización
    def optimize(self):
        # inicializa el registro e ingresa al ciclo principal hasta que alcance el número máximo de iteraciones

        # revisión de parámetros, si no tiene las dimensiones, dominio o función de costo, marca un error
        if(self.num_var == 0):
            print("Error, first set the number of variables and their boundaries")
        elif(self.cost_function == None):
            print("Error, first define the cost function to be used")
        else:
            # inicializa el registro mediante un sampleo aleatorio, respetando las restricciones de cada variable
            pop = np.zeros((self.pop_size, self.num_var + 1))
            w = np.zeros(self.k)

            for i in range(self.k):
                for j in range(self.num_var):
                    # inicializa el registro de soluciones de manera aleatoria
                    self.SA[i, j] = np.random.uniform(
                        self.var_ranges[j][0], self.var_ranges[j][1])
                # obtiene el costo inicial para cada solución
                self.SA[i, -1] = self.cost_function(self.SA[i, 0:self.num_var])
            # ordena el registro de soluciones (la mejor solución primero)
            self.SA = self.SA[self.SA[:, -1].argsort()]

            x = np.linspace(1, self.k, self.k)
            # los pesos se toman como una función de rango gaussiana (función de densidad de probabilidad)
            w = norm.pdf(x, 1, self.q*self.k)
            # probabilidades de seleccionar soluciones como guías de búsqueda
            p = w/sum(w)

            # el algoritmo se ejecuta hasta llegar al número máximo de iteraciones
            #fig = plt.figure()
            for iteration in range(self.max_iter):
                self.minima_array[0].append(self.SA[0, 0])
                self.minima_array[1].append(self.SA[1, 1])
                self.iteration_array.append(iteration)
                # print("Iteration: {} | Best Fitness: {}".format(
                #    iteration, self.SA[0, 0]))

                Mi = self.SA[:, 0:self.num_var]
                # por cada hormiga de la población:
                for ant in range(self.pop_size):
                    # selecciona una solución del registro de soluciones para probar en función de las probabilidades de p
                    l = self._biased_selection(p)

                    # calcula la desviación estándar de todas las variables de la solución l
                    for var in range(self.num_var):
                        sigma_sum = 0
                        for i in range(self.k):
                            sigma_sum += abs(self.SA[i, var] - self.SA[l, var])
                        sigma = self.xi * (sigma_sum/(self.k - 1))

                        # prueba de la distribución normal con la media Mi and desviación estándar sigma
                        pop[ant, var] = np.random.normal(Mi[l, var], sigma)

                        # verifica que no ocurra una violación del espacio utilizando la estrategia de posición aleatoria
                        if(pop[ant, var] < self.var_ranges[var][0] or pop[ant, var] > self.var_ranges[var][1]):
                            pop[ant, var] = np.random.uniform(
                                self.var_ranges[var][0], self.var_ranges[var][1])

                    # evalúa el costo de la nueva solución
                    pop[ant, -1] = self.cost_function(pop[ant, 0:self.num_var])

                # agrega nuevas soluciones al registro
                self.SA = np.append(self.SA, pop, axis=0)
                # ordena el registro de soluciones de acuerdo a la aptitud de cada solución
                self.SA = self.SA[self.SA[:, -1].argsort()]
                # elimina las peores soluciones
                self.SA = self.SA[0:self.k, :]

            self.best_solution = self.SA[0, :]
            return(self.best_solution)


def _fitness(x):
    y = x*np.sin(np.sqrt(abs(x)))
    return(y)


def main():
    colony = ant_colony()
    ranges = [[-500, 500]]

    colony.set_cost(schwefel)
    colony.set_variables(1, ranges)
    colony.set_parameters(100, 5, 50, 0.01, 0.85)

    solution = colony.optimize()

    #fitness = np.vectorize(_fitness)
    #x = np.linspace(start=-512, stop=512, num=200)

    #plt.plot(x, fitness(x), 'm--')
    #plt.scatter(solution[0], solution[0], marker='*', c='r', s=200)
    #plt.suptitle('Schwefel Function', fontsize=12)

    #f, axarr = plt.subplots(2)
    #axarr[0].plot(colony.iteration_array, colony.minima_array[1], c='r')
    #axarr[0].set_title('Evolution of Optimization', fontsize=12)
    #axarr[0].set_ylabel('Global Minima', fontsize=9)
    #axarr[1].plot(colony.iteration_array, colony.minima_array[0], c='g')
    #axarr[1].set_ylabel('Function', fontsize=9)
    #plt.xlabel('Iteration', fontsize=10)

    # print("Best solution: {}".format(solution))
    return(schwefel([solution[0], solution[0]]))
