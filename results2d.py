import pickle
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pygmo import fast_non_dominated_sorting, hypervolume
import numpy as np
import pandas as pd
import datetime as dt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


def main(numL, numC):
    # name = 'zdt1'
    # name = 'zdt2'
    # name = 'zdt3'
    # name = 'zdt6'
    # name = 'zdt4'
    # name = 'mw1'
    # name = 'mw2'
    # name = 'mw3'
    # name = 'mw4'
    # name = 'mw5'
    # name = 'mw6'
    # name = 'mw7'
    # problem = get_problem(name)

    name = 'wfg1'
    problem = get_problem(name, n_var = 6, n_obj =2)

    # name = 'zdt6'
    # problem = get_problem(name, normalize=False)

    # problem = get_problem("mw1", n_var =10, n_obj=2)
    pf_a = problem.pareto_front()
    # pf_a = problem.pareto_front(use_cache=False)

    # f = open('results.pkl', 'rb')
    # f = open('testes/gecco/201124/results_11_300sim_30iter_128pop_-1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/201124/results_12_300sim_30iter_128pop_-1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/201124/results_13_300sim_30iter_128pop_-1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/101224/results_14_300sim_100iter_128pop_-1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/201124/results_16_300sim_30iter_128pop_-1.0alpha_4060.pkl', 'rb')

    # f = open('testes/gecco/101224/results_31_100sim_30iter_128pop_-1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/101224/results_33_100sim_30iter_128pop_-1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/101224/results_35_100sim_30iter_128pop_-1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/101224/results_37_100sim_30iter_128pop_-1.0alpha_4060.pkl', 'rb')

    # zdts 64 bits - hiper continuaram a mesma coisa(zdt4 melhorou) e o tempo aumentou um pouco
    # f = open('testes/gecco/171224/results_11_300sim_30iter_128pop_1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/171224/results_12_300sim_30iter_128pop_1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/171224/results_13_300sim_30iter_128pop_1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/171224/results_14_100sim_100iter_128pop_1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/171224/results_16_300sim_30iter_128pop_1.0alpha_4060.pkl', 'rb')

    # mw1 - abaixou o tempo e hiper ficou diferente na quarta casa decimal
    # f = open('testes/gecco/171224/results_31_100sim_30iter_128pop_1.0alpha_4060.pkl', 'rb')
    # ganhou do mw1 da cpu
    # f = open('testes/gecco/171224/results_31_100sim_40iter_128pop_1.0alpha_4060.pkl', 'rb')
    # baixou tempo e ganhou da cpu
    # f = open('testes/gecco/171224/results_33_100sim_30iter_128pop_1.0alpha_4060.pkl', 'rb')
    # continua ganhando da cpu e tempo quase igual
    # f = open('testes/gecco/171224/results_35_100sim_30iter_128pop_1.0alpha_4060.pkl', 'rb')
    #contiunua perdendo
    # f = open('testes/gecco/171224/results_37_100sim_40iter_128pop_1.0alpha_4060.pkl', 'rb')

    # f = open('testes/gecco/201224/results_32_100sim_30iter_128pop_1.0alpha_4060.pkl', 'rb')
    # f = open('testes/gecco/201224/results_36_100sim_30iter_128pop_1.0alpha_4060.pkl', 'rb')

    #testes 200625
    # f = open('testes/results_11_100sim_30iter_128pop_1.0alpha_3060.pkl', 'rb')
    # f = open('testes/results_12_100sim_30iter_128pop_1.0alpha_3060.pkl', 'rb')
    # f = open('testes/results_13_100sim_30iter_128pop_1.0alpha_3060.pkl', 'rb')
    # f = open('testes/results_16_100sim_30iter_128pop_1.0alpha_3060.pkl', 'rb')
    # f = open('testes/200625/results_14_100sim_30iter_128pop_1.0alpha_3060.pkl', 'rb')
    # f = open('results_21_100sim_30iter_128pop_6posdim_1.0alpha_3060_alan.pkl', 'rb')
    # f = open('results_21_1000sim_60iter_128pop_6posdim_1.0alpha_3060.pkl', 'rb')
    # f = open('results_21_100sim_600iter_128pop_6posdim_1.0alpha_3060.pkl', 'rb')
    # f = open('results_21_50sim_1200iter_128pop_6posdim_1.0alpha_3060.pkl', 'rb')
    # f = open('results_31_100sim_50iter_128pop_3posdim_1.0alpha_3060.pkl', 'rb')
    # f = open('results_32_100sim_600iter_128pop_15posdim_0.1alpha_3060.pkl', 'rb')
    f = open('results_21_50sim_1200iter_128pop_6posdim_1.0alpha_3060.pkl', 'rb')

    results = pickle.load(f)
    f.close()

    pos = []
    fit = []
    lenMem = []

    figsize = (10, 10)
    if numL > 0:
        for i in range(numL):
            for j in range(numC):
                numT = i*numC+j
                result = results[numT]
                lenMem.append(result[2][0])

                fim = 256 + result[2][0]
                fit.extend(result[1][256:fim])
                # fit.extend(result[1][:128])

                if numL * numC <= lim:
                    plt.subplot(numL, numC, numT + 1)
                    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', result[1][256:fim, 0],
                         result[1][256:fim, 1], 'bo')
        plt.show()
    else:
        for i in range(numC):
            result = results[i]
            lenMem.append(result[2][0])
            fit.extend(result[1][256:])
            fim = 256 + result[2][0]
            a, b, c, d = fast_non_dominated_sorting(points=result[1][256:])

    f.close()
    fit = np.array(fit)

    plt.figure()
    plt.title('all memory points ')
    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit[:, 0], fit[:, 1], 'bo')
    plt.legend(['paretto', 'GPU'])
    plt.show()

    a, b, c, d = fast_non_dominated_sorting(points=fit)
    fit = fit[a[0]]

    x = np.max(fit[:, 0]) + 0.1
    y = np.max(fit[:, 1]) + 0.1
    x2 = np.max(pf_a[:, 0]) + 0.1
    y2 = np.max(pf_a[:, 1]) + 0.1
    x = max(x, x2)
    y = max(y, y2)
    ref = [x, y]
    # ref = [2]*2

    print('ref', ref)

    plt.figure()
    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit[:, 0], fit[:, 1], 'bo')
    plt.title('after fast non dominating sorting')
    plt.legend(['pareto', 'GPU'])
    plt.show()

    plt.figure()
    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro')
    plt.title(name.upper()+ ' PARETO')
    plt.show()

    plt.figure()
    plt.plot(fit[:, 0], fit[:, 1], 'ro')
    plt.title(name.upper()+ ' MESH GPU')
    plt.show()

    hv = hypervolume(pf_a)
    print('hypervolume_paretto', hv.compute(ref))

    hv2 = hypervolume(fit)
    print('hypervolume_gpu', hv2.compute(ref))

    gpu_pareto = abs(hv.compute(ref) - hv2.compute(ref))
    print('hypervolume_gpu_pareto', gpu_pareto)

    gpu = results['gpu']
    plt.figure()
    plt.title(name.upper()+ ' GPU times')
    df = pd.DataFrame(gpu)
    df.boxplot()
    print('GPU\n', df.describe())
    plt.show()

    print('total time in GPU: ', sum(gpu))
    print('total time in GPU: ', str(dt.timedelta(seconds=sum(gpu)))[:-7])


if __name__ == '__main__':
    numL = 50
    numC = 1
    lim = 2
    main(numL, numC)