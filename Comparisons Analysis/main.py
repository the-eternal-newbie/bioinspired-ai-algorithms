from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from tabulate import tabulate

import aco_schwefel
import ais_schwefel
import de_schwefel
import ga_schwefel
import pso_schwefel
import abc_schwefel


def mann_whitney(data1, data2):
    # compare samples
    stat, p = mannwhitneyu(data1, data2)
    print("[Mann-Whitney] Valor de p = {}".format(p))
    # interpret
    alpha = 0.05
    if(p > alpha):
        print('El algoritmo a no es mejor que el algoritmo b (no refuta la hipotesis H0)')
    else:
        print('[MEJOR ALGORITMO] (refuta la hipotesis H0)')
    return(p)


def wilcoxon_ranked(data1, data2):
    stat, p = wilcoxon(data1, data2)
    print("[Wilcoxon] Valor de p = {}".format(p))
    # interpret
    alpha = 0.05
    if(p > alpha):
        print('El algoritmo a no es mejor que el algoritmo b (no refuta la hipotesis H0)')
    else:
        print('[MEJOR ALGORITMO] (refuta la hipotesis H0)')
    return(p)


def kruskal_wallis(data1, data2):
    # compare samples
    stat, p = kruskal(data1, data2)
    print("[Wilcoxon] Valor de p = {}".format(p))
    # interpret
    alpha = 0.05
    if(p > alpha):
        print('El algoritmo a no es mejor que el algoritmo b (no refuta la hipotesis H0)')
    else:
        print('[MEJOR ALGORITMO] (refuta la hipotesis H0)')
    return(p)


if __name__ == "__main__":

    abc = []
    aco = []
    ais = []
    de = []
    ga = []
    pso = []

    for _ in range(20):
        abc.append(abc_schwefel.main())
        aco.append(aco_schwefel.main())
        ais.append(ais_schwefel.main())
        de.append(de_schwefel.main())
        ga.append(ga_schwefel.main())
        pso.append(pso_schwefel.main())

    abc_row = ["ABC", "*", "*", "*", "*", "*", "*"]
    aco_row = ["ACO", "*", "*", "*", "*", "*", "*"]
    ais_row = ["AIS", "*", "*", "*", "*", "*", "*"]
    de_row = ["DE", "*", "*", "*", "*", "*", "*"]
    ga_row = ["GA", "*", "*", "*", "*", "*", "*"]
    pso_row = ["PSO", "*", "*", "*", "*", "*", "*"]

    abc_row[2] = mann_whitney(abc, aco)
    abc_row[3] = wilcoxon_ranked(abc, ais)
    abc_row[4] = kruskal_wallis(abc, de)
    abc_row[5] = mann_whitney(abc, ga)
    abc_row[6] = wilcoxon_ranked(abc, pso)

    aco_row[1] = abc_row[2]
    aco_row[3] = kruskal_wallis(aco, ais)
    aco_row[4] = mann_whitney(aco, de)
    aco_row[5] = wilcoxon_ranked(aco, ga)
    aco_row[6] = kruskal_wallis(aco, pso)

    ais_row[1] = abc_row[3]
    ais_row[2] = aco_row[3]
    ais_row[4] = mann_whitney(ais, de)
    ais_row[5] = wilcoxon_ranked(ais, ga)
    ais_row[6] = kruskal_wallis(ais, pso)

    de_row[1] = abc_row[4]
    de_row[2] = aco_row[4]
    de_row[3] = ais_row[4]
    de_row[5] = mann_whitney(de, ga)
    de_row[6] = wilcoxon_ranked(de, pso)

    ga_row[1] = abc_row[5]
    ga_row[2] = aco_row[5]
    ga_row[3] = ais_row[5]
    ga_row[4] = de_row[5]
    ga_row[6] = kruskal_wallis(ga, pso)

    pso_row[1] = abc_row[6]
    pso_row[2] = aco_row[6]
    pso_row[3] = ais_row[6]
    pso_row[4] = de_row[6]
    pso_row[5] = ga_row[6]

    print(tabulate([abc_row, aco_row, ais_row, de_row, ga_row, pso_row], headers=[
          'ABC', 'ACO', 'AIS', 'DE', 'GA', 'PSO']))
