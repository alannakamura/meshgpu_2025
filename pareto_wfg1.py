from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems.many.wfg import WFG1
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np
import pickle
import pygmo as pg
from matplotlib.pyplot import plot

l = []
dim =2
c = np.empty((0, 2))
f = open('wfg1', 'wb')

# Problema
problem = WFG1(n_var=12, n_obj=dim, k=4)

# Parada
termination = get_termination("n_gen", 30)

s = Scatter(title="NSGA-II on WFG1")
for i in range(30):
    # Algoritmo
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=20, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    # Otimização
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=i,
        verbose=True
    )

    l.append((res.F, res.X))
    s.add(l[-1][0])
    c = np.concatenate((c, res.F), axis=0)
    pass

fronts = pg.fast_non_dominated_sorting(points=c)[0]
c = c[fronts[0]]
s.add(c, marker='s',color='red')
pickle.dump(l, f)
f.close()
# Plot

# s = Scatter(title="NSGA-II on WFG1")
# s.add(l[0][0], color='red', marker = 'o')
# s.add(l[1][0], color = 'blue', marker='x')
# s.add(l[2][0], color = 'black',marker='s')
# s.add(l[3][0], color='green', marker = 'o')
# s.add(l[4][0], color = 'magenta', marker='x')
# s.add(l[5][0], color = 'pink',marker='s')
# s.show()

# s.add(l[0][0], color='red', marker = 'o')
# s.add(l[1][0], color = 'blue', marker='x')
# s.add(l[2][0], color = 'black',marker='s')
# s.add(l[3][0], color='green', marker = 'o')
# s.add(l[4][0], color = 'magenta', marker='x')
# s.add(l[5][0], color = 'pink',marker='s')
s.show()
