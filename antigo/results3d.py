import pickle
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pygmo import fast_non_dominated_sorting, hypervolume
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
import numpy as np
import pandas as pd
import datetime as dt
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.optimize import minimize


def main(numL, numC, test_nsga2):

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=100)
    #dtlz1 a dtlz4
    name = 'dtlz1'
    problem = get_problem(name)
    pf_a = problem.pareto_front(ref_dirs)

    # name = 'dtlz7'
    # problem = get_problem(name)
    # pf_a = problem.pareto_front()

    print('gpu', problem)

    f = open('results_1_100sim_30iter_128pop_10posdim_1.0alpha_3060.pkl', 'rb')
    results = pickle.load(f)
    f.close()


    # pos = []
    fit = []
    fit3 = []
    lenMem = []

    figsize = (10, 10)
    if numL > 0:
        for i in range(numL):
            for j in range(numC):
                numT = i*numC+j
                result = results[numT]
                lenMem.append(result[2][0])
                inicio = int(len(result[1])*2/3)
                fim = int(len(result[1])*2/3) + result[2][0]
                fit2 = result[1][inicio:fim]

                # inf1 = np.where(np.sum(np.isposinf(fit2), 1) > 0)[0]
                # print('pos', inf1)
                # inf2 = np.where(np.sum(np.isneginf(fit2), 1) > 0)[0]
                # print('neg', inf2)

                fit.extend(fit2)
                fit3.append(fit2)
                if numL*numC <= lim:
                    plt.subplot(numL, numC, numT + 1)
                    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro',
                            fit2[:, 0], fit2[:, 1], 'bo')
        if numL * numC < lim:
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

    fit4 = fit3[0]
    for i in fit3[1:]:
        fit4 = np.concatenate((fit4, i), axis=0)
        a, b, c, d = fast_non_dominated_sorting(points=fit4)
        fit4 = fit4[a[0]]


    #teste nsga2
    # sim = 1
    # res2 = []
    # print(problem)
    # for i in range(sim):
    #     # problem = get_problem("dtlz1")
    #     algorithm = NSGA2(pop_size=128)
    #     res = minimize(problem,
    #                    algorithm,
    #                    ('n_gen', 200),
    #                    seed=i,
    #                    verbose=False)
    #     res2.extend(res.F.tolist())
    #     pass
    #
    # res2 = np.array(res2)
    # a, b, c, d = fast_non_dominated_sorting(points=res2)
    # res2 = res2[a[0]]


    x = np.max(fit4[:, 0]) + 0.1
    y = np.max(fit4[:, 1]) + 0.1
    z = np.max(fit4[:, 2]) + 0.1
    x2 = np.max(pf_a[:, 0]) + 0.1
    y2 = np.max(pf_a[:, 1]) + 0.1
    z2 = np.max(pf_a[:, 2]) + 0.1
    # x3 = np.max(res2[:, 0]) + 0.1
    # y3 = np.max(res2[:, 1]) + 0.1
    # z3 = np.max(res2[:, 2]) + 0.1
    # x = max(x, x2, x3)
    # y = max(y, y2, y3)
    # z = max(z, z2, z3)
    x = max(x, x2)
    y = max(y, y2)
    z = max(z, z2)
    ref = [x, y, z]
    # ref = [2]*3
    # ref[2] = 7
    # ref = [60] * 3 #dtlz1
    # ref = [91] * 3 #dtlz3

    print('ref', ref)

    hv = hypervolume(pf_a)
    print('hypervolume_paretto', hv.compute(ref))

    hv2 = hypervolume(fit4)
    print('hypervolume_mesh', hv2.compute(ref))

    # hv3 = hypervolume(res2)
    # print('hypervolume_nsga2', hv3.compute(ref))

    # print('hypervolume_ratio_mp', hv2.compute(ref) / hv.compute(ref))
    # print('hypervolume_ratio_np', hv3.compute(ref) / hv.compute(ref))
    # print('hypervolume_ratio_mn', hv2.compute(ref) / hv3.compute(ref),'\n')
    hypervolume_distance_mesh_pareto = abs(hv2.compute(ref) - hv.compute(ref))
    print('hypervolume_distance_mesh_pareto', hypervolume_distance_mesh_pareto)
    # hypervolume_distance_nsga2_pareto = abs(hv3.compute(ref) - hv.compute(ref))
    # print('hypervolume_distance_nsga2_pareto', hypervolume_distance_nsga2_pareto)
    # if hypervolume_distance_mesh_pareto < hypervolume_distance_nsga2_pareto:
    #     print('melhor e MESH')
    # elif hypervolume_distance_mesh_pareto > hypervolume_distance_nsga2_pareto:
    #     print('melhor e NSGA2')
    # else:
    #     print('iguais')
    # print('hypervolume_distance_mesh_nsga2', abs(hv2.compute(ref) - hv3.compute(ref)), '\n')

    plt.figure()
    plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', fit4[:, 0], fit4[:, 1], 'bo')
    plt.title('after fast non dominating sorting MESH')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['paretto', 'GPU'])
    # plt.axis([0, 1.5, 0, 1.5])
    plt.show()

    # if test_nsga2:
    #     plt.figure()
    #     plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', res2[:, 0], res2[:, 1], 'bo')
    #     plt.title('after fast non dominating sorting NSGA2')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.legend(['paretto', 'GPU'])
    #     plt.show()

    plt.figure()
    plt.plot(pf_a[:, 0], pf_a[:, 2], 'ro', fit4[:, 0], fit4[:, 2], 'bo')
    plt.title('after fast non dominating sorting MESH')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend(['paretto', 'GPU'])
    # plt.axis([0, 1.5, 0, 1.5])
    plt.show()

    # if test_nsga2:
    #     plt.figure()
    #     plt.plot(pf_a[:, 0], pf_a[:, 2], 'ro', res2[:, 0], res2[:, 2], 'bo')
    #     plt.title('after fast non dominating sorting NSGA2')
    #     plt.xlabel('x')
    #     plt.ylabel('z')
    #     plt.legend(['paretto', 'GPU'])
    #     plt.show()

    plt.figure()
    plt.plot(pf_a[:, 1], pf_a[:, 2], 'ro', fit4[:, 1], fit4[:, 2], 'bo')
    plt.title('after fast non dominating sorting MESH')
    plt.xlabel('y')
    plt.ylabel('z')
    plt.legend(['paretto', 'GPU'])
    # plt.axis([0, 1.5, 0, 1.5])
    plt.show()

    # if test_nsga2:
    #     plt.figure()
    #     plt.plot(pf_a[:, 1], pf_a[:, 2], 'ro', res2[:, 1], res2[:, 2], 'bo')
    #     plt.title('after fast non dominating sorting NSGA2')
    #     plt.xlabel('y')
    #     plt.ylabel('z')
    #     plt.legend(['paretto', 'GPU'])
    #     plt.show()

    s = Scatter(angle=(20, 20), title = 'MESH 3D')
    s.add(pf_a, color='red')
    # s.add(fit[a[0]], color = 'blue')
    s.add(fit4, color='blue')
    s.show()

    s = Scatter(angle=(20, 20), title = name.upper()+' PARETO')
    s.add(pf_a, color='red')
    s.show()

    s = Scatter(angle=(20, 20), title = name.upper()+' MESH GPU')
    s.add(fit4, color='red')
    s.show()

    gpu = results['gpu']
    plt.figure()
    plt.title(name.upper()+' GPU TIMES')
    df = pd.DataFrame(gpu)
    df.boxplot()
    print('GPU\n', df.describe())
    plt.show()
    print('total time in GPU: ', sum(gpu))
    print('total time in GPU: ', str(dt.timedelta(seconds=sum(gpu)))[:-7])

    # if test_nsga2:
    #     s = Scatter(angle=(20, 20), title='NSGA2 3D')
    #     s.add(pf_a, color='red')
    #     # s.add(fit[a[0]], color = 'blue')
    #     s.add(res2, color='blue')
    #     s.show()

    # plt.figure()
    # plt.title('lenMem')
    # plt.plot(np.arange(len(lenMem)), lenMem, 'ro')
    # plt.show()


if __name__ == '__main__':
    numL = 100
    numC = 1
    lim = 2
    test_nsga2 = True
    main(numL, numC, test_nsga2)