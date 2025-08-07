import pickle
from matplotlib.pyplot import *
from pymoo.problems.many.wfg import WFG2
from pygmo import hypervolume

f = open('wfg1.pkl', 'rb')
a = pickle.load(f)
# show()
f.close()

dim = 2
problem = WFG2(n_var=12, n_obj=dim, k=4)
pymoo_par = problem.pareto_front()

plot(a[:,0], a[:,1],'ro', pymoo_par[:,0], pymoo_par[:,1], 'bo')

ref = 5,5
hv1 = hypervolume(pymoo_par)
res1 = hv1.compute(ref)
print('hypervolume pymoo', res1)

hv2 = hypervolume(a)
res2 = hv2.compute(ref)
print('hypervolume manual', res2)

print('difference', abs(res2-res1))

show()