# Declaración de librerías necesarias para graficar
# y hacer operaciones matemáticas
from mpl_toolkits import mplot3d
from matplotlib import cm
import random
import numpy as np
import matplotlib.pyplot as plt


# Función de prueba
def schwefel(values):
    """ Dominio de la función: -500 <= xi <= 500 (i = 1, ..., d).
        Mínimo global: f(x*) = 0, at x* = (420.9687, ..., 420.9687) """
    xy = np.asarray_chkfinite(values)
    n = len(xy)
    return(418.9829*n - sum(xy * np.sin(np.sqrt(abs(xy)))))


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
        xyz = (round(data[0], 4), round(data[1], 4))
        # Si los parámetros contienen más de un elemento (puntos), significa
        # que se está haciendo un recorrido de las mejores partículas
        # por lo que se hace un coloreado distinto en cada situación
        if(len(arr) > 1):
            def r(): return random.randint(0, 255)
            color = '#%02X%02X%02X' % (r(), r(), r())
            size = 40
            ax.set_title('Schwefel Sine - Evolution of global minima')
        else:
            color = 'r'
            size = 60
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


# Clase partícula
class Particle():

    '''
    Controla los valores internos de la partícula y su información,
    además de calcular por sí sola el movimiento de esta
    '''

    def __init__(self):
        # Arreglo de aleatorios dentro del rango de valores de la función [-500, 500]
        self.position = np.asarray_chkfinite([(-1)**(bool(random.getrandbits(1))) * random.random()*500,
                                              (-1)**(bool(random.getrandbits(1))) * random.random()*500])
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.asarray_chkfinite([0, 0])

    def __str__(self):
        return(self.pbest_position)

    # Función individual del movimiento de la partícula
    def move(self):
        self.position = self.position + self.velocity


# Clase espacio de búsqueda
class Space():

    '''
    Es responsable de mantener todas las partículas, identificar y establecer los mejores
    valores de posición de todas las partículas, gestionar los criterios del marge de error,
    calcular el mejor global y establecer la mejor posición.
    '''

    def __init__(self, target, target_error, n_particles):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = np.asarray_chkfinite(
            [random.random()*500, random.random()*500])

    # Asignación de la mejor posición global
    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = schwefel(particle.position)
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position

    # Asignación del mejor valor global
    def set_gbest(self):
        for particle in self.particles:
            best_fitness_cadidate = schwefel(particle.position)
            if(self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position

    # Rutina de movimiento de las partículas
    def move_particles(self):
        for particle in self.particles:
            global W

            # Fórmula de la velocidad (afectada por inercia y por coeficientes de aceleración)
            new_velocity = (W*particle.velocity) + (c1*random.random()) * (particle.pbest_position - particle.position) + \
                (random.random()*c2) * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()


if __name__ == "__main__":
    # Definición de los factores de inercia y aceleración
    W = 0.5
    c1 = 0.2
    c2 = 0.4

    # Asignación de los parámetros del algoritmo (número de iteraciones,
    # margen de error y el número de partículas)
    n_iterations = 100
    target_error = 1e-8
    n_particles = 100

    search_space = Space(1, target_error, n_particles)
    particles_vector = [Particle() for _ in range(search_space.n_particles)]
    search_space.particles = particles_vector
    global_bests = []

    iteration = 0
    while(iteration < n_iterations):
        search_space.set_pbest()
        search_space.set_gbest()
        global_bests.append(
            (search_space.gbest_position[0], search_space.gbest_position[1]))

        # Si el margen de error ha sido alcanzado (es el mínimo permitido),
        # el algoritmo se detiene, puesto que es muy complicado que llegue
        # a 0 la minimización
        if(abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
            break

        search_space.move_particles()
        iteration += 1

    print("The best solution is: {}, {} with a minima of {}".format(round(search_space.gbest_position[0], 4),
                                                                    round(search_space.gbest_position[1], 4), schwefel(search_space.gbest_position)))

    # Gráficas de las funciones y comportamiento del algoritmo
    plotConvergence(global_bests)
    plotFunction(global_bests)
    plotFunction([search_space.gbest_position])

    plt.show()
