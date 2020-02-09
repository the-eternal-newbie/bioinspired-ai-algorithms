import numpy as np
import matplotlib.pyplot as plt

# se define la función objetivo (schwefel sine)
def _fitness(x):
    y = x*np.sin(np.sqrt(abs(x)))
    return(y)

# función para determinar el padre más aptop de una generación
def _get_fittest_parent(parents, fitness):
    # de cada set de padres, se evalúa en la función objetivo
    _fitness = fitness(parents)
    PFitness = list(zip(parents, _fitness))
    # se crea una lista la cual se ordena con los valores de los padres
    # en función de la función objetivo
    PFitness.sort(key=lambda x: x[1], reverse=True)
    # el primer elemento de dicha lista será el mejor padre con el valor óptimo
    best_parent, best_fitness = PFitness[0]
    return(round(best_parent, 4), round(best_fitness, 4))

# función de mutación
def mutate(parents, fitness_function):
    n = int(len(parents))
    scores = fitness_function(parents)
    # acepta sólo valores positivos (buscando la maximización de la función)
    idx = scores > 0
    scores = scores[idx]
    parents = np.array(parents)[idx]
    # se realiza un remuestreo de los padres con probabilidades proporcionales a la aptitud
    # luego, agrega algo de ruido para obtener una mutación 'aleatoria'
    children = np.random.choice(parents, size=n, p=scores / scores.sum())
    children = children + np.random.uniform(-0.51, 0.51, size=n)
    return(children.tolist())


def GA(parents, fitness_function, popsize=100, max_iter=100):
    History = []
    # Inicialización de los padres: gen cero
    best_parent, best_fitness = _get_fittest_parent(
        parents, fitness)  # se obtiene el individuo más apto
    print("Generation {} | Best fitness {} | Current Fitness {} | Current Parent {}".format(
        0, best_fitness, best_fitness, best_parent))

    # Primer plotting ded los padres iniciales
    x = np.linspace(start=-20, stop=20, num=200)  # population range
    plt.plot(x, fitness_function(x))
    plt.scatter(parents, fitness_function(parents), marker='x')

    # por cada generación
    for i in range(1, max_iter):
        parents = mutate(parents, fitness_function=fitness_function)
        curr_parent, curr_fitness = _get_fittest_parent(
            parents, fitness_function)  # se obtiene el individuo más apto

        # actualización de los valores más aptos
        if(curr_fitness > best_fitness):
            best_fitness = curr_fitness
            best_parent = curr_parent

        curr_parent, curr_fitness = _get_fittest_parent(
            parents, fitness_function)
        if(i % 10 == 0):
            print("Generation {} | Best Fitness {} | Current Fitness {} | Current Parent".format(
                i, best_fitness, curr_fitness, curr_parent))
        # se almacena el máximo óptimo de cada generación
        History.append((i, np.max(fitness_function(parents))))

    plt.scatter(parents, fitness_function(parents))
    plt.scatter(best_parent, fitness_function(
        best_parent), marker='*', c='r', s=200)
    plt.pause(0.09)
    plt.ioff()
    # regresa los padres
    print("Generation {} | Best fitness {} | Best Parent {}".format(
        i, best_fitness, best_parent))

    return best_parent, best_fitness, History


if __name__ == "__main__":

    fitness = np.vectorize(_fitness)

    x = np.linspace(start=-500, stop=500, num=200)
    plt.plot(x, fitness(x))

    x = np.linspace(start=-500, stop=500, num=200)
    init_pop = np.random.uniform(low=-500, high=500, size=200)

    parent_, fitness_, history_ = GA(init_pop, fitness)
    # se genera una gráfica con el historial de las generaciones
    # del algoritmo con los valores más óptimos de cada generación; 
    fig = plt.figure()
    for i in history_:
        plt.plot(i[0], i[1], marker='.', c='r')
        fig.suptitle('Evolution of Optimization', fontsize=12)
        plt.xlabel('Generation', fontsize=9)
        plt.ylabel('Best Fitness', fontsize=9)
        
    print("Top Parent {} | Top Fitness {}".format(parent_, fitness_))
