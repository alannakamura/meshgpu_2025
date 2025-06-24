import pickle
from tqdm import tqdm
import os
import pycuda.driver as drv

# problem = [21]
problem = [11,12,13,14,16]
# problem = [1,2,3,4,5,6,7]
# problem = [31, 33, 35, 36, 37]
alpha = [1.0]*len(problem)
for j in range(len(problem)):
    print('problem', problem[j])

    GPU = '3060'
    # GPU = '4060'

    num = 100
    iterations = 30
    population = 128
    pos_dim = 10

    f = open('results.pkl', 'wb')
    results = {'count': -1, 'cpu': [], 'gpu': []}
    pickle.dump(results, f)
    f.close()

    for i in tqdm(range(num)):
        print('simulation ', i+1, (i + 1) / num * 100, '%')
        os.system("python run2.py "+str(problem[j]) + ' ' +
                  str(iterations)+ ' ' + str(population) + ' '+ str(alpha[j])+ ' '+
                  str(pos_dim))

    alpha2 = str(alpha[j]).split('.')
    os.rename('results.pkl', 'results_' + str(problem[j]) + '_'
              + str(num) +'sim_'
              + str(iterations) +'iter_'
              + str(population) +'pop_'
              + str(pos_dim) +'posdim_'
              + alpha2[0] + '.' + alpha2[1] +'alpha_'
              + GPU +'.pkl')


