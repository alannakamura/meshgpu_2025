###########################################################################
# Lucas Braga, MS.c. (email: lucas.braga.deo@gmail.com )
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# Carolina Marcelino, PhD (email: carolimarc@ic.ufrj.br)
# June 16, 2021
###########################################################################
# Copyright (c) 2021, Lucas Braga, Gabriel Matos Leite, Carolina Marcelino
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS USING 
# THE CREATIVE COMMONS LICENSE: CC BY-NC-ND "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# import numpy as np
import datetime
import sys

from objectives import *
# from MESH import *
from MESH import *
import pygmo as pg
import pickle
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import itertools
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    problem = int(sys.argv[1])
    max_num_iters = int(sys.argv[2])
    population = int(sys.argv[3])
    alpha = float(sys.argv[4])
    pos_dim = int(sys.argv[5])
    print('\npos_dim', pos_dim)
else:
    problem = 36
    max_num_iters = 1
    population = 128
    alpha = -1.0
    pos_dim = 10

    f = open('results2.pkl', 'wb')
    results = {'count': -1, 'cpu': [], 'gpu': []}
    pickle.dump(results, f)
    f.close()

optimizationMap = {
    1: 'DTLZ1',
    2: 'DTLZ2',
    3: 'DTLZ3',
    4: 'DTLZ4',
    5: 'DTLZ5',
    6: 'DTLZ6',
    7: 'DTLZ7',
    11: 'ZDT1',
    12: 'ZDT2',
    13: 'ZDT3',
    14: 'ZDT4',
    15: 'ZDT5',
    16: 'ZDT6',
    21: 'WFG1',
    22: 'WFG2',
    23: 'WFG3',
    24: 'WFG4',
    25: 'WFG5',
    26: 'WFG6',
    27: 'WFG7',
    28: 'WFG8',
    29: 'WFG9',
    30: 'f',
    31: 'MW1',
    32: 'MW2',
    33: 'MW3',
    34: 'MW4',
    35: 'MW5',
    36: 'MW6',
    37: 'MW7',
    38: 'MW8',
    39: 'MW9',
    310: 'MW10'
}


def main():
    Path("result").mkdir(parents=False, exist_ok=True)
    
    # for func_n in [1, 2, 3, 5, 6, 7]:
    for func_n in [int(problem)]:
        func_name = optimizationMap[func_n]
        # num_runs = 30
        num_runs = 1
        if (11 <= func_n <= 16
                or 21 <=func_n <= 21
                or 31 <= func_n <= 33
                or 35 <= func_n <=37
                or 39 <=func_n <= 310):
            objectives_dim = 2
        elif 1 <= func_n <= 7 or func_n == 34:
            objectives_dim = 3
        else:
            objectives_dim = -1
        func = get_function(func_n, objectives_dim)

        otimizations_type = [False] * objectives_dim
        max_iterations = 0
        # max_fitness_eval = 15000
        max_fitness_eval = -1
        # max_num_iters = 100
        # position_dim = 10
        position_dim = pos_dim
        if func_n in [1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 16,
                      31, 32, 33, 34, 35, 36, 37, 39, 310]:
            position_max_value = [1] * position_dim
            # position_min_value = [1e-6] * position_dim
            position_min_value = [0] * position_dim
        if func_n in [14]:
            position_max_value = [10] * position_dim
            position_min_value = [-10] * position_dim
            position_max_value[0] = 1
            position_min_value[0] = 0
        if func_n in [15]:
            position_max_value = [2**5-1] * position_dim
            position_min_value = [0] * position_dim
            position_max_value[0] = 2**30-1
        if func_n == 21:
            position_max_value = [0] * position_dim
            position_min_value = [0] * position_dim
            for i in range(position_dim):
                position_max_value[i] = 2*(i+1)

        # population_size = 100
        population_size = population
        # num_final_solutions = 128  # aparece no final
        memory_size = population
        memory_update_type = 0

        communication_probability = 0.7 #0.5
        mutation_rate = 0.9
        personal_guide_array_size = 3

        global_best_attribution_type = 1     # 0 -> E1 | 1 -> E2 | 2 -> E3 | 3 -> E4
        Xr_pool_type = 1                  # 0 ->  V1 | 1 -> V2 | 2 -> V3
        DE_mutation_type = 0        # 0 -> DE\rand\1\Bin | 1 -> DE\rand\2\Bin | 2 -> DE/Best/1/Bin | 3 -> DE/Current-to-best/1/Bin | 4 -> DE/Current-to-rand/1/Bin
        crowd_distance_type = 1     # 0 -> Crowding Distance Tradicional | 1 -> Crowding Distance Suganthan


        config = f"E{global_best_attribution_type + 1}V{Xr_pool_type + 1}D{DE_mutation_type + 1}C{crowd_distance_type+1}"
        
        print(f"Running E{global_best_attribution_type+1}V{Xr_pool_type+1}D{DE_mutation_type+1}C{crowd_distance_type+1} on {optimizationMap[func_n]}")

        print('problem', problem, optimizationMap[problem], 'max_iters', max_num_iters, 'alpha', alpha)

        for i in tqdm(range(num_runs)):
            params = MESH_Params(objectives_dim, otimizations_type, max_iterations,
                                 max_fitness_eval, position_dim, position_max_value,
                                 position_min_value, population_size, memory_size,
                                 memory_update_type, global_best_attribution_type,
                                 DE_mutation_type, Xr_pool_type, crowd_distance_type,
                                 communication_probability, mutation_rate, personal_guide_array_size,
                                 func_n=func_n,
                                 gpu=True)
            # MCDEEPSO = MESH(params, func)
            MCDEEPSO = MESH(params, func, max_num_iters=max_num_iters, alpha=alpha)
            MCDEEPSO.log_memory = f"result/{config}C1_{i}-{optimizationMap[func_n]}-{objectives_dim}obj-"

            cpu, gpu = MCDEEPSO.run(func_name,False, False)

            f = open('results.pkl', 'rb')
            results = pickle.load(f)
            f.close()

            f = open('results.pkl', 'wb')
            results['cpu'].append(sum(cpu))
            results['gpu'].append(sum(gpu))
            pickle.dump(results, f)
            f.close()

        # ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=combined)
        # n = num_final_solutions
        # if len(ndf[0]) < num_final_solutions:
        #     n = len(ndf[0])
        # best_idx = pg.sort_population_mo(points=combined)[:n]
        #
        # result['combined'] = (best_idx, combined[best_idx])
        #
        # with open(f'result/{config}_{optimizationMap[func_n]}_{objectives_dim}obj.pkl', 'wb') as f:
        #     pickle.dump(result, f)


if __name__ == '__main__':
    main()
