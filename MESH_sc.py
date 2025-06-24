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
import os
# import numpy as np
import sys
import copy

import numpy as np
from scipy.stats import truncnorm
from Particle import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.curandom import *
# import cupy as cp
import struct
from datetime import datetime as dt
import sys
import pickle
from pycuda.tools import clear_context_caches

# a = np.zeros(9)

class MESH_Params:
    def __init__(self,
                 objectives_dim,
                 otimizations_type,
                 max_iterations,
                 max_fitness_eval,
                 position_dim,
                 position_max_value,
                 position_min_value,
                 population_size,
                 memory_size,
                 memory_update_type,
                 global_best_attribution_type,
                 DE_mutation_type,
                 Xr_pool_type,
                 crowd_distance_type,
                 communication_probability,
                 mutation_rate,
                 personal_guide_array_size,
                 secondary_params=False,
                 initial_state=False,
                 func_n = 11,
                 gpu=False):

        self.objectives_dim = objectives_dim
        self.otimizations_type = otimizations_type

        self.max_iterations = max_iterations
        self.max_fitness_eval = max_fitness_eval

        self.position_dim = position_dim
        self.position_max_value = position_max_value
        self.position_min_value = position_min_value

        self.velocity_min_value = list()
        self.velocity_max_value = list()
        for i in range(position_dim):
            self.velocity_min_value.append(-1 * self.position_max_value[i] + self.position_min_value[i])
            self.velocity_max_value.append(-1 * self.velocity_min_value[i])

        self.population_size = population_size

        self.memory_size = memory_size
        self.memory_update_type = memory_update_type

        self.global_best_attribution_type = global_best_attribution_type

        self.DE_mutation_type = DE_mutation_type
        self.Xr_pool_type = Xr_pool_type
        self.crowd_distance_type = crowd_distance_type

        self.communication_probability = communication_probability
        self.mutation_rate = mutation_rate

        self.personal_guide_array_size = personal_guide_array_size

        self.secondary_params = secondary_params
        self.initial_state = initial_state

        self.func_n = func_n

        if gpu:
            self.objectives_dim_g = (
                cuda.mem_alloc(np.array(objectives_dim, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.objectives_dim_g,
                             np.array(self.objectives_dim, dtype=np.int32))

            self.otimizations_type_g = (
                cuda.mem_alloc(np.array(otimizations_type, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.otimizations_type_g,
                             np.array(self.otimizations_type, dtype=np.int32))

            self.max_iterations_g = (
                cuda.mem_alloc(np.array(max_iterations, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.max_iterations_g,
                             np.array(self.max_iterations, dtype=np.int32))
            self.max_fitness_eval_g = (
                cuda.mem_alloc(np.array(max_fitness_eval, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.max_fitness_eval_g,
                             np.array(self.max_fitness_eval, dtype=np.int32))

            self.position_dim_g = (
                cuda.mem_alloc(np.array(position_dim, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.position_dim_g,
                             np.array(self.position_dim, dtype=np.int32))

            self.position_max_value_g = (
                cuda.mem_alloc(np.array(position_max_value, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.position_max_value_g,
                             np.array(self.position_max_value, dtype=np.float64))

            self.position_min_value_g = (
                cuda.mem_alloc(np.array(position_min_value, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.position_min_value_g,
                             np.array(self.position_min_value, dtype=np.float64))

            self.velocity_min_value_g = (
                cuda.mem_alloc(np.array(self.velocity_min_value, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.velocity_min_value_g,
                             np.array(self.velocity_min_value, dtype=np.float64))

            self.velocity_max_value_g = (
                cuda.mem_alloc(np.array(self.velocity_max_value, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.velocity_max_value_g,
                             np.array(self.velocity_max_value, dtype=np.float64))

            self.population_size_g = (
                cuda.mem_alloc(np.array(self.population_size, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.population_size_g,
                             np.array(self.population_size, dtype=np.int32))

            self.memory_size_g = (
                cuda.mem_alloc(np.array(self.memory_size, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.memory_size_g,
                             np.array(self.memory_size, dtype=np.int32))

            self.current_memory_size = 0

            self.current_memory_size_g = (
                cuda.mem_alloc(np.array(self.memory_size, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.current_memory_size_g,
                             np.array(-1, dtype=np.int32))

            self.memory_update_type_g = (
                cuda.mem_alloc(np.array(self.memory_update_type, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.memory_update_type_g,
                             np.array(self.memory_update_type, dtype=np.int32))

            self.global_best_attribution_type_g = (
                cuda.mem_alloc(np.array(self.global_best_attribution_type, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.global_best_attribution_type_g,
                             np.array(self.global_best_attribution_type, dtype=np.int32))

            self.DE_mutation_type_g = (
                cuda.mem_alloc(np.array(self.DE_mutation_type, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.DE_mutation_type_g,
                             np.array(self.DE_mutation_type, dtype=np.int32))

            self.Xr_pool_type_g = (
                cuda.mem_alloc(np.array(self.Xr_pool_type, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.Xr_pool_type_g,
                             np.array(self.Xr_pool_type, dtype=np.int32))

            self.crowd_distance_type_g = (
                cuda.mem_alloc(np.array(self.crowd_distance_type, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.crowd_distance_type_g,
                             np.array(self.crowd_distance_type, dtype=np.int32))

            self.communication_probability_g = (
                cuda.mem_alloc(np.array(self.communication_probability, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.communication_probability_g,
                             np.array(self.communication_probability, dtype=np.float64))

            self.mutation_rate_g = (
                cuda.mem_alloc(np.array(self.mutation_rate, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.mutation_rate_g,
                             np.array(self.mutation_rate, dtype=np.float64))

            self.personal_guide_array_size_g = (
                cuda.mem_alloc(np.array(self.personal_guide_array_size, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.personal_guide_array_size_g,
                             np.array(self.personal_guide_array_size, dtype=np.int32))

            self.secondary_params_g = (
                cuda.mem_alloc(np.array(self.secondary_params, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.secondary_params_g,
                             np.array(self.secondary_params, dtype=np.int32))

            self.initial_state_g = (
                cuda.mem_alloc(np.array(self.initial_state, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.initial_state_g,
                             np.array(self.initial_state, dtype=np.int32))

            self.func_n_g = (
                cuda.mem_alloc(np.array([0], dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.func_n_g,np.array([self.func_n], dtype=np.int32))

            fmt = "P" * 21
            data = struct.pack(fmt, self.objectives_dim_g, self.otimizations_type_g,
                               self.max_iterations_g, self.max_fitness_eval_g, self.position_dim_g,
                               self.position_max_value_g, self.position_min_value_g,
                               self.velocity_min_value_g, self.velocity_max_value_g,
                               self.population_size_g, self.memory_size_g, self.memory_update_type_g,
                               self.global_best_attribution_type_g, self.DE_mutation_type_g,
                               self.Xr_pool_type_g, self.crowd_distance_type_g,
                               self.communication_probability_g, self.mutation_rate_g,
                               self.personal_guide_array_size_g, self.secondary_params_g,
                               self.initial_state_g)
            size = struct.calcsize(fmt)
            # print('size', size)
            self.gpu = cuda.mem_alloc(size)
            cuda.memcpy_htod(self.gpu, data)

            # f = open('mesh.cu')
            # code = f.read()
            # f.close()
            # mod = SourceModule(code)
            #
            # test_mesh_params = mod.get_function("test_mesh_params")
            # test_mesh_params(self.gpu, block=(1, 1, 1), grid=(1, 1))
            # cuda.Context.synchronize()

            # cuda.memcpy_htod(self.c_gpu, data)


class MESH:
    def __init__(self, params, fitness_function, max_num_iters=40, alpha=-1, gpu=True):
        self.params = params
        self.stopping_criteria_reached = False
        self.generation_count = 0

        self.population = []
        self.population_copy = []

        self.memory = []
        self.fronts = []

        self.fitness_function = fitness_function
        self.fitness_eval_count = 0

        self.max_num_iters = max_num_iters
        self.alpha = alpha

        # np.random.seed(0)
        w1 = np.random.uniform(0.0, 1.0, [4, self.params.population_size])
        w2 = np.random.uniform(0.0, 0.5, [1, self.params.population_size])
        w3 = np.random.uniform(0.0, 2.0, [1, self.params.population_size])

        self.weights = np.concatenate((w1, w2, w3), axis=0)
        self.weights_copy = []

        self.update_from_differential_mutation = False
        self.log_memory = False
        self.copy_pop = True
        self.gpu = gpu

        if gpu:
            total = self.params.population_size * 2 + self.params.memory_size
            # total2 = self.params.population_size*3+self.params.memory_size

            self.position = (
                cuda.mem_alloc(np.zeros(total * self.params.position_dim, dtype=np.float64).nbytes))

            # self.velocity = (
            #     cuda.mem_alloc(np.zeros(total*self.params.position_dim, dtype=np.float64).nbytes))

            self.fitness = (
                cuda.mem_alloc(np.zeros(total * self.params.objectives_dim, dtype=np.float64).nbytes))

            self.alpha_g = (
                cuda.mem_alloc(np.array(self.alpha, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.alpha_g,
                             np.array(self.alpha, dtype=np.float64))

            self.stopping_criteria_reached_g = (
                cuda.mem_alloc(np.array(self.stopping_criteria_reached, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.stopping_criteria_reached_g,
                             np.array(self.stopping_criteria_reached, dtype=np.int32))

            self.generation_count_g = (
                cuda.mem_alloc(np.array(self.generation_count, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.generation_count_g,
                             np.array(self.generation_count, dtype=np.int32))

            self.fitness_eval_count_g = (
                cuda.mem_alloc(np.array(self.fitness_eval_count, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.fitness_eval_count_g,
                             np.array(self.fitness_eval_count, dtype=np.int32))

            self.weights_g = (
                cuda.mem_alloc(np.array(self.weights, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.weights_g,
                             np.array(self.weights, dtype=np.float64))

            self.weights_copy_g = (
                cuda.mem_alloc(np.array(self.weights, dtype=np.float64).nbytes))

            self.update_from_differential_mutation2 = np.zeros(self.params.population_size * 2 +
                                                               self.params.memory_size, dtype=np.int32)
            self.update_from_differential_mutation_g = (
                cuda.mem_alloc(self.update_from_differential_mutation2.nbytes))
            cuda.memcpy_htod(self.update_from_differential_mutation_g,
                             self.update_from_differential_mutation2)

            self.copy_pop_g = (
                cuda.mem_alloc(np.array(self.copy_pop, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.copy_pop_g,
                             np.array(self.copy_pop, dtype=np.int32))

            self.domination_counter_g = (
                cuda.mem_alloc(np.zeros((total+1) * (total+1), dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.domination_counter_g,
                             np.zeros((total+1) * (total+1), dtype=np.int32))

            self.fronts_g = (
                cuda.mem_alloc(np.zeros(2 * self.params.population_size, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.fronts_g,
                             np.zeros(2 * self.params.population_size, dtype=np.int32))

            self.front0_mem_g = (
                cuda.mem_alloc(np.zeros(2*self.params.population_size + self.params.memory_size,
                                        dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.front0_mem_g,
                             np.zeros(2*self.params.population_size + self.params.memory_size,
                                      dtype=np.int32))

            self.tam_front0_mem_g = (
                cuda.mem_alloc(np.zeros(1, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.tam_front0_mem_g, np.zeros(1, dtype=np.int32))

            self.tams_fronts_g = (
                cuda.mem_alloc(np.zeros(2 * self.params.population_size, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.tams_fronts_g,
                             np.zeros(2 * self.params.population_size, dtype=np.int32))

            self.crowding_distance_g = (
                cuda.mem_alloc(np.zeros(total, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.crowding_distance_g,
                             np.zeros(total, dtype=np.float64))

            # total = (self.params.population_size * 2 + self.params.memory_size)*self.
            # params.personal_guide_array_size
            # self.personal_best_g = (
            #     cuda.mem_alloc(np.zeros(total, dtype=np.int32).nbytes))
            # cuda.memcpy_htod(self.personal_best_g,
            #                  -1+np.zeros(total, dtype=np.int32))

            self.whatPersonal = -1 * np.ones(self.params.population_size * 2 + self.params.memory_size,
                                             dtype=np.int32)
            self.whatPersonal_g = cuda.mem_alloc(self.whatPersonal.nbytes)
            cuda.memcpy_htod(self.whatPersonal_g, self.whatPersonal)

            # self.xr_pool = np.zeros(self.params.population_size * 2 + self.params.memory_size,
            #                         dtype=np.int32)
            # self.xr_pool_g = cuda.mem_alloc(self.xr_pool.nbytes)
            # cuda.memcpy_htod(self.xr_pool_g, self.xr_pool)

            self.xr_pool = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
                                    self.params.population_size,
                                    dtype=np.int32)
            self.xr_pool_g = cuda.mem_alloc(self.xr_pool.nbytes)
            cuda.memcpy_htod(self.xr_pool_g, self.xr_pool)

            self.xr_list = np.zeros((self.params.population_size * 2 + self.params.memory_size) * 5,
                                    dtype=np.int32)
            self.xr_list_g = cuda.mem_alloc(self.xr_list.nbytes)
            cuda.memcpy_htod(self.xr_list_g, self.xr_list)

            self.xst = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
                                self.params.position_dim, dtype=np.float64)
            self.xst_g = cuda.mem_alloc(self.xst.nbytes)
            cuda.memcpy_htod(self.xst_g, self.xst)

            self.xst_fitness = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
                                        self.params.objectives_dim, dtype=np.float64)
            self.xst_fitness_g = cuda.mem_alloc(self.xst_fitness.nbytes)
            cuda.memcpy_htod(self.xst_fitness_g, self.xst_fitness)

            self.mutation_index = np.zeros((self.params.population_size * 2 + self.params.memory_size)
                                           , dtype=np.int32)
            self.mutation_index_g = cuda.mem_alloc(self.mutation_index.nbytes)
            cuda.memcpy_htod(self.mutation_index_g, self.mutation_index)

            self.mutation_chance = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
                                            self.params.position_dim, dtype=np.float64)
            self.mutation_chance_g = cuda.mem_alloc(self.mutation_chance.nbytes)
            cuda.memcpy_htod(self.mutation_chance_g, self.mutation_chance)

            self.xst_dominate = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
                                         1, dtype=np.int32)
            self.xst_dominate_g = cuda.mem_alloc(self.xst_dominate.nbytes)
            cuda.memcpy_htod(self.xst_dominate_g, self.xst_dominate)

            self.personal_best_tam = np.ones((self.params.population_size * 2 + self.params.memory_size)
                                             , dtype=np.int32)
            self.personal_best_tam_g = cuda.mem_alloc(self.personal_best_tam.nbytes)
            cuda.memcpy_htod(self.personal_best_tam_g, self.personal_best_tam)

            self.front0 = np.zeros(total+1, dtype=np.float64)
            self.front0_g = cuda.mem_alloc(self.front0.nbytes)
            cuda.memcpy_htod(self.front0_g, self.front0)

            self.tam_front0 = np.zeros(1, dtype=np.int32)
            self.tam_front0_g = cuda.mem_alloc(self.tam_front0.nbytes)
            cuda.memcpy_htod(self.tam_front0_g, self.tam_front0)

            self.communication = -1 * np.ones(total * self.params.position_dim, dtype=np.float64)
            self.communication_g = cuda.mem_alloc(self.communication.nbytes)
            cuda.memcpy_htod(self.communication_g, self.communication)

            self.cooperation_rand = -1 * np.ones(total, dtype=np.float64)
            self.cooperation_rand_g = cuda.mem_alloc(self.cooperation_rand.nbytes)
            cuda.memcpy_htod(self.cooperation_rand_g, self.cooperation_rand)

            self.population_index = np.zeros(self.params.population_size * 2, dtype=np.int32)
            self.population_index_g = cuda.mem_alloc(self.population_index.nbytes)

            self.aux = np.zeros((2 * self.params.population_size + self.params.memory_size) *
                                self.params.position_dim, dtype=np.float64)
            self.aux_g = cuda.mem_alloc(self.aux.nbytes)

            self.aux2 = np.zeros((2 * self.params.population_size + self.params.memory_size) *
                                 self.params.position_dim, dtype=np.float64)
            self.aux2_g = cuda.mem_alloc(self.aux2.nbytes)

            self.aux3 = np.zeros((2 * self.params.population_size + self.params.memory_size) *
                                 self.params.position_dim * self.params.personal_guide_array_size, dtype=np.float64)
            self.aux3_g = cuda.mem_alloc(self.aux3.nbytes)

            self.aux4 = np.zeros(2*self.params.population_size*self.params.objectives_dim,
                                 dtype=np.int32)
            self.aux4_g = cuda.mem_alloc(self.aux4.nbytes)

            self.initial_memory = np.zeros(self.params.population_size, dtype=np.int32)
            self.initial_memory_g = cuda.mem_alloc(self.initial_memory.nbytes)

            fmt = "P" * 9
            data = struct.pack(fmt, self.params.gpu, self.stopping_criteria_reached_g, self.generation_count_g,
                               self.fitness_eval_count_g, self.weights_g, self.weights_copy_g,
                               self.update_from_differential_mutation_g, self.copy_pop_g,
                               self.domination_counter_g)
            size = struct.calcsize(fmt)
            # print('\nsize\n', size)
            self.gpu = cuda.mem_alloc(size)
            cuda.memcpy_htod(self.gpu, data)

            f = open('mesh.cu')
            code = f.read()
            f.close()
            self.mod = SourceModule(code, no_extern_c=True)

            # test_mesh = self.mod.get_function("test_mesh")
            # test_mesh(self.gpu, block=(1, 1, 1), grid=(1, 1))
            # cuda.Context.synchronize()

    def init_population(self):
        # for i in range(self.params.population_size):
            # new_particle = Particle(self.params.position_min_value, self.params.position_max_value,
            #                         self.params.position_dim,
            #                         self.params.velocity_min_value, self.params.velocity_max_value,
            #                         self.params.objectives_dim, self.params.otimizations_type,
            #                         self.params.secondary_params, self.gpu)
            # new_particle.init_random()
            #
            # self.population.append(new_particle)
        if self.gpu:
            total = self.params.population_size * 2 + self.params.memory_size
            # total2 = self.params.population_size * 3 + self.params.memory_size
            # self.crowding_distance_g = (
            #     cuda.mem_alloc(np.zeros(total, dtype=np.float64).nbytes))
            # cuda.memcpy_htod(self.crowding_distance_g,
            #                  np.zeros(total, dtype=np.float64))
            self.rank_g = (
                cuda.mem_alloc(np.zeros(total, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.rank_g,
                             -1 * np.ones(total, dtype=np.int32))

            self.fitness_g = (
                cuda.mem_alloc(np.zeros(total * self.params.objectives_dim, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.fitness_g,
                             np.zeros(total * self.params.objectives_dim, dtype=np.float64))

            # self.position_g = (
            #     cuda.mem_alloc(np.zeros(total2 * self.params.position_dim, dtype=np.float64).nbytes))
            # position = []
            # for i in self.population:
            #     position.extend(i.position)
            # position = np.array(position, dtype=np.float64)
            # cuda.memcpy_htod(self.position_g, position)


            self.seed_g = cuda.mem_alloc(np.zeros(1, dtype=np.float64).nbytes)
            cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))

            self.position_g = (
                cuda.mem_alloc(np.zeros(total*self.params.position_dim, dtype=np.float64).nbytes))
            init_population = self.mod.get_function("init_population")

            init_population(self.position_g, self.params.position_dim_g,self.seed_g,
                            self.params.position_min_value_g, self.params.position_max_value_g,
                       block=(int(self.params.population_size), 1, 1), grid=(1, 1, 1))
            cuda.Context.synchronize()

            #teste
            # teste_p = np.zeros(384 * 10, dtype=np.float64)
            # cuda.memcpy_dtoh(teste_p, self.position_g)
            # teste_p.shape = 384, 10
            # print(teste_p[0])

            self.velocity_g = (
                cuda.mem_alloc(np.zeros(total * self.params.position_dim, dtype=np.float64).nbytes))
            # velocity = []
            # for i in self.population:
            #     velocity.extend(i.velocity)
            # velocity = np.array(velocity, dtype=np.float64)
            # cuda.memcpy_htod
            init_population = self.mod.get_function("init_population")
            cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))
            init_population(self.velocity_g, self.params.position_dim_g, self.seed_g,
                            self.params.velocity_min_value_g, self.params.velocity_max_value_g,
                            block=(int(self.params.population_size), 1, 1), grid=(1, 1, 1))
            cuda.Context.synchronize()

            # teste
            # teste_p = np.zeros(384 * 10, dtype=np.float64)
            # cuda.memcpy_dtoh(teste_p, self.position_g)
            # teste_p.shape = 384, 10
            # print(teste_p)
            # pass

            #teste
            # teste_v = np.zeros(384 * 10, dtype=np.float64)
            # cuda.memcpy_dtoh(teste_v, self.velocity_g)
            # teste_v.shape = 384, 10
            # print(teste_v[0])
            # pass

            # self.velocity_g = rand((self.params.population_size * self.params.position_dim,1),
            #                        np.float64)

            # self.domination_counter_g = (
            #     cuda.mem_alloc(np.zeros(total+1, dtype=np.int32).nbytes))
            # cuda.memcpy_htod(self.domination_counter_g,
            #                  np.zeros(total+1, dtype=np.int32))

            self.dominated_set_g = (
                cuda.mem_alloc(np.zeros(total * total, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.dominated_set_g,
                             np.zeros(self.params.population_size, dtype=np.int32))

            self.sigma_g = (
                cuda.mem_alloc(np.zeros(total * self.params.objectives_dim, dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.sigma_g,
                             np.zeros(total * self.params.objectives_dim, dtype=np.float64))

            self.global_best_g = (
                cuda.mem_alloc(np.zeros(total, dtype=np.int32).nbytes))
            cuda.memcpy_htod(self.global_best_g,
                             np.zeros(total, dtype=np.int32))

            self.personal_best_position_g = (
                cuda.mem_alloc(np.zeros(total * self.params.position_dim * self.params.personal_guide_array_size,
                                        dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.personal_best_position_g,
                             1e10 + np.zeros(total * self.params.position_dim * self.params.personal_guide_array_size,
                                             dtype=np.float64))

            self.personal_best_velocity_g = (
                cuda.mem_alloc(np.zeros(total * self.params.position_dim * self.params.personal_guide_array_size,
                                        dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.personal_best_velocity_g,
                             np.zeros(total * self.params.position_dim * self.params.personal_guide_array_size,
                                      dtype=np.float64))

            self.personal_best_fitness_g = (
                cuda.mem_alloc(np.zeros(total * self.params.objectives_dim * self.params.personal_guide_array_size,
                                        dtype=np.float64).nbytes))
            cuda.memcpy_htod(self.personal_best_fitness_g,
                             np.zeros(total * self.params.objectives_dim * self.params.personal_guide_array_size,
                                      dtype=np.float64))

            self.population_copy2 = copy.deepcopy(self.population)

            # pass
            # f = open('mesh.cu')
            # code = f.read()
            # f.close()
            # mod = SourceModule(code)
            #
            # test_position = self.mod.get_function("test_position")
            # test_position(self.position_g, self.params.position_dim_g, block=(1, 1, 1), grid=(1, 1))
            # cuda.Context.synchronize()
            # pass

    def particle_copy(self, particle):
        copy = Particle(self.params.position_min_value, self.params.position_max_value, self.params.position_dim,
                        self.params.velocity_min_value, self.params.velocity_max_value,
                        self.params.objectives_dim, self.params.otimizations_type,
                        self.params.secondary_params)
        copy.position = particle.position
        copy.fitness = particle.fitness
        copy.velocity = particle.velocity
        if particle.personal_best is not None:
            copy_personal_best_list = []
            for pb in particle.personal_best:
                copy_pb = Particle(self.params.position_min_value, self.params.position_max_value,
                                   self.params.position_dim,
                                   self.params.velocity_min_value, self.params.velocity_max_value,
                                   self.params.objectives_dim, self.params.otimizations_type,
                                   self.params.secondary_params)
                copy_pb.position = pb.position
                copy_pb.fitness = pb.fitness
                copy_personal_best_list.append(copy_pb)
            copy.personal_best = copy_personal_best_list
        return copy

    def fitness_evaluation(self, function, *args):
        self.fitness_eval_count = self.fitness_eval_count + 1
        if self.params.initial_state:
            args_and_initial_state = []
            args_and_initial_state.append(args[0])
            args_and_initial_state.extend(self.params.initial_state)
            return function(args_and_initial_state)
        else:
            return function(*args)

    def fast_nondominated_sort_gpu(self, first_front_only=False, use_copy_population=False,
                                   specific_population=None):
        population = []
        population2 = []
        fronts = [[]]

        for p in self.population:
            population.append(p)
            population2.append(p.position)
        if use_copy_population:
            for o in self.population_copy:
                population.append(o)
                population2.append(o.position)
        population2 = np.array(population2, dtype=np.float64)
        domination = np.zeros((len(population), len(population)), dtype=np.int32)
        # counter = np.zeros((population.shape[0]), dtype=np.float64)

        population_g = cuda.mem_alloc(population2.nbytes)
        # counter_g = cuda.mem_alloc(counter.nbytes)
        dim_g = cuda.mem_alloc(np.array(population2.shape, dtype=np.int32).nbytes)
        domination_g = cuda.mem_alloc(domination.nbytes)

        cuda.memcpy_htod(population_g, population2)
        # cuda.memcpy_htod(counter_g, counter)
        cuda.memcpy_htod(dim_g, np.array(population2.shape, dtype=np.int32))
        cuda.memcpy_htod(domination_g, domination)

        f = open('testes/ordenacao.cu')
        code = f.read()
        f.close()
        mod = SourceModule(code)

        sort2 = mod.get_function("sort2")

        block_x = 32
        block_y = 32
        grid_x = int(len(population) / 32)
        grid_y = int(len(population) / 32)
        sort2(population_g, dim_g, domination_g, block=(block_x, block_y, 1), grid=(grid_x, grid_y, 1))
        cuda.Context.synchronize()

        # cuda.memcpy_dtoh(population, population_g)
        cuda.memcpy_dtoh(domination, domination_g)

        for i in range(len(population)):
            self.population[i].domination_counter = np.sum(domination[:, i])
            self.population[i].dominated_set = np.array(self.population)[domination[i, :] == 1]
            if self.population[i].domination_counter == 0:
                fronts[0].append(self.population[i])
                self.population[i].rank = 0

        i = 0
        if not first_front_only:
            while len(fronts[i]) != 0:
                new_front = []
                for p in fronts[i]:
                    for s in p.dominated_set:
                        s.domination_counter = s.domination_counter - 1
                        if s.domination_counter == 0:
                            new_front.append(s)
                            s.rank = i + 1
                i += 1
                fronts.append(list(new_front))
            fronts.pop()
            for p in population:
                p.dominated_set = []
            return fronts
        else:
            for p in population:
                p.dominated_set = []
            return fronts[0]

    def fast_nondominated_sort(self, first_front_only=False, use_copy_population=False,
                               specific_population=None):
        population = []
        fronts = []
        fronts.append([])

        if specific_population != None:
            for s in specific_population:
                population.append(s)
        else:
            for p in self.population:
                population.append(p)
            if use_copy_population:
                for o in self.population_copy:
                    population.append(o)

        for p in population:
            p.domination_counter = 0
            for q in population:
                if p == q:
                    continue
                if p >> q:
                    p.dominated_set.append(q)
                elif p << q:
                    p.domination_counter = p.domination_counter + 1
            if p.domination_counter == 0:
                fronts[0].append(p)
                p.rank = 0

        i = 0
        if not first_front_only:
            while len(fronts[i]) != 0:
                new_front = []
                for p in fronts[i]:
                    for s in p.dominated_set:
                        s.domination_counter = s.domination_counter - 1
                        if s.domination_counter == 0:
                            new_front.append(s)
                            s.rank = i + 1
                i += 1
                fronts.append(list(new_front))
            fronts.pop()
            for p in population:
                p.dominated_set = []
            return fronts
        else:
            for p in population:
                p.dominated_set = []
            return fronts[0]

    def crowding_distance(self, front):
        for j in front:
            j.crowd_distance = 0
        for objective_index in range(self.params.objectives_dim):
            front.sort(key=lambda x: x.fitness[objective_index])
            front[0].crowd_distance = sys.maxsize
            front[-1].crowd_distance = sys.maxsize
            for p in range(1, len(front) - 1):
                if front[p].crowd_distance == sys.maxsize:
                    continue
                if front[-1].fitness[objective_index] - front[0].fitness[objective_index] == 0:
                    continue
                front[p].crowd_distance += (front[p + 1].fitness[objective_index] - front[p - 1].fitness[
                    objective_index]) / (front[-1].fitness[objective_index] - front[0].fitness[objective_index])

            # apenas para validacao, apagar depois. Foi necessario para manter a ordem identica
            # dos fronts
            # if len(front) == 256:
            #     for i in range(len(front)):
            #         if len(np.where(np.array(self.population_copy2) == front[i])[0]) != 0:
            #             self.aux4[objective_index*256+i] = (
            #                 np.where(np.array(self.population_copy2) == front[i])[0][0])
            #         else:
            #             self.aux4[objective_index * 256 + i] = ((
            #                 np.where(np.array(self.population_copy3) == front[i])[0][0])
            #                                                     + self.params.population_size)
            # pass



    def crowd_distance_selection(self, particle_A, particle_B):
        if particle_A.rank < particle_B.rank:
            return particle_A
        elif particle_B.rank < particle_A.rank:
            return particle_B
        elif particle_A.rank == particle_B.rank:
            if particle_A.crowd_distance > particle_B.crowd_distance:
                return particle_A
            elif particle_B.crowd_distance >= particle_A.crowd_distance:
                return particle_B

    def check_position_limits(self, position_input):
        position = position_input[:]
        for i in range(self.params.position_dim):
            if position[i] < self.params.position_min_value[i]:
                position[i] = self.params.position_min_value[i]
            if position[i] > self.params.position_max_value[i]:
                position[i] = self.params.position_max_value[i]
        return position

    def check_velocity_limits(self, velocity_input, position_input=None):
        velocity = velocity_input[:]
        if position_input is not None:
            position = position_input[:]
            for i in range(self.params.position_dim):
                if position[i] == self.params.position_min_value[i] and velocity[i] < 0:
                    velocity[i] = -1 * velocity[i]
                elif position[i] == self.params.position_max_value[i] and velocity[i] > 0:
                    velocity[i] = -1 * velocity[i]
        else:
            for i in range(self.params.position_dim):
                if velocity[i] < self.params.velocity_min_value[i]:
                    velocity[i] = self.params.velocity_min_value[i]
                if velocity[i] > self.params.velocity_max_value[i]:
                    velocity[i] = self.params.velocity_max_value[i]
        return velocity

    def euclidian_distance(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return np.linalg.norm(a - b)

    def sigma_eval(self, particle):
        squared_power = np.power(particle.fitness, 2)
        denominator = np.sum(squared_power)
        numerator = []
        # testeTempo250624 para self.params.objectives_dim > 3?
        if self.params.objectives_dim == 2:
            numerator = squared_power[0] - squared_power[1]
        else:
            for i in range(self.params.objectives_dim):
                if i != self.params.objectives_dim - 1:
                    numerator.append(squared_power[i] - squared_power[i + 1])
                else:
                    numerator.append(squared_power[i] - squared_power[0])
        sigma = np.divide(numerator, denominator)
        particle.sigma_value = sigma
        pass
        # particle.sigma_value = sigma.astype(np.float64)
        pass

    def sigma_nearest(self, particle, search_pool):
        sigma_distance = sys.maxsize
        nearest_particle = None
        for p in search_pool:
            if particle != p:
                new_distance = self.euclidian_distance(particle.sigma_value, p.sigma_value)
                if sigma_distance > new_distance:
                    sigma_distance = new_distance
                    nearest_particle = p
        if nearest_particle is None:  # se a distancia inicial testeTempo250624 maxima, entao
            # sempre vai ter um melhor pq
            # 1 distancia nao sera menor que a maxima?
            nearest_particle = particle
        nearest_particle = copy.deepcopy(nearest_particle)
        # entao global best nao testeTempo250624 a melhor posicao entre todas as particulas?
        particle.global_best = nearest_particle

    def move_particle(self, particle, particle_index, is_copy):
        if is_copy:
            weights = self.weights_copy
        else:
            weights = self.weights

        # original
        # personal_best_pos = particle.personal_best[np.random.choice(len(particle.personal_best))].position

        # teste
        personal_best_pos = particle.personal_best[np.random.choice(len(particle.personal_best))]
        if is_copy:
            self.whatPersonal[particle_index + self.params.population_size] = (
                np.where(np.array(particle.personal_best) == personal_best_pos))[0][0]
        else:
            self.whatPersonal[particle_index] = (
                np.where(np.array(particle.personal_best) == personal_best_pos))[0][0]
        personal_best_pos = personal_best_pos.position

        inertia_term = np.asarray(particle.velocity) * weights[0][particle_index]
        memory_term = weights[1][particle_index] * (np.asarray(personal_best_pos) - np.asarray(particle.position))

        # original
        # communication = (np.random.uniform(0.0, 1.0, self.params.position_dim) < self.params.communication_probability) * 1

        # teste
        communication = (np.random.uniform(0.0, 1.0, self.params.position_dim))
        if is_copy:
            for i in range(self.params.position_dim):
                self.communication[(particle_index + self.params.population_size) * self.params.position_dim + i] \
                    = communication[i]
        else:
            for i in range(self.params.position_dim):
                self.communication[particle_index * self.params.position_dim + i] \
                    = communication[i]
        communication = communication < self.params.communication_probability * 1

        # original
        # nao entendi o por que de multiplicar o global best por 1+(entre 0 testeTempo250624 1)*w3
        # cooperation_term = weights[2][particle_index] * (np.asarray(particle.global_best.position)
        # * (1 + (weights[3][particle_index] * np.random.normal(0, 1))) - np.asarray(particle.position))
        # cooperation_term = cooperation_term * communication

        # teste
        if is_copy:
            self.cooperation_rand[particle_index + self.params.population_size] = np.random.normal(0, 1)
            rand = self.cooperation_rand[particle_index + self.params.population_size]
        else:
            self.cooperation_rand[particle_index] = np.random.normal(0, 1)
            rand = self.cooperation_rand[particle_index]
        cooperation_term = weights[2][particle_index] * (np.asarray(particle.global_best.position)
                                                         * (1 + (
                        weights[3][particle_index] * rand)) - np.asarray(particle.position))
        cooperation_term = cooperation_term * communication

        # original
        new_velocity = inertia_term + memory_term + cooperation_term
        new_velocity = self.check_velocity_limits(new_velocity)

        new_position = np.asarray(particle.position) + new_velocity
        new_position = self.check_position_limits(new_position)
        new_velocity = self.check_velocity_limits(new_velocity, new_position)

        particle.velocity = new_velocity
        particle.position = new_position

        if self.params.secondary_params:
            fit_eval = self.fitness_evaluation(self.fitness_function, particle.position)
            particle.fitness = fit_eval[0]
            particle.secondary_params = fit_eval[1:]
        else:
            particle.fitness = self.fitness_evaluation(self.fitness_function, particle.position)

    def mutate_weights(self):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                # weight[i] = weight[i] + np.random.normal(0,1)*self.params.mutation_rate
                if i < 4:
                    self.weights[i][j] = truncnorm.rvs(0, 1) * self.params.mutation_rate
                    if self.weights[i][j] > 1:
                        self.weights[i][j] = 1
                    elif self.weights[i][j] < 0:
                        self.weights[i][j] = 0
                if i == 4:
                    self.weights[i][j] = truncnorm.rvs(0, 0.5) * self.params.mutation_rate
                    if self.weights[i][j] > 0.5:
                        self.weights[i][j] = 0.5
                    elif self.weights[i][j] < 0:
                        self.weights[i][j] = 0
                if i == 5:
                    self.weights[i][j] = truncnorm.rvs(0, 2) * self.params.mutation_rate
                    if self.weights[i][j] > 2:
                        self.weights[i][j] = 2
                    elif self.weights[i][j] < 0:
                        self.weights[i][j] = 0
        if self.copy_pop:
            for i in range(len(self.weights_copy)):
                for j in range(len(self.weights_copy[i])):
                    # weight[i] = weight[i] + np.random.normal(0,1)*self.params.mutation_rate
                    if i < 4:
                        self.weights_copy[i][j] = truncnorm.rvs(0, 1) * self.params.mutation_rate
                        if self.weights_copy[i][j] > 1:
                            self.weights_copy[i][j] = 1
                        elif self.weights_copy[i][j] < 0:
                            self.weights_copy[i][j] = 0
                    if i == 4:
                        self.weights_copy[i][j] = truncnorm.rvs(0, 0.5) * self.params.mutation_rate
                        if self.weights_copy[i][j] > 0.5:
                            self.weights_copy[i][j] = 0.5
                        elif self.weights_copy[i][j] < 0:
                            self.weights_copy[i][j] = 0
                    if i == 5:
                        self.weights_copy[i][j] = truncnorm.rvs(0, 2) * self.params.mutation_rate
                        if self.weights_copy[i][j] > 2:
                            self.weights_copy[i][j] = 2
                        elif self.weights_copy[i][j] < 0:
                            self.weights_copy[i][j] = 0

    def differential_mutation(self, particle, particle_index):
        Xr_pool = []

        # np.random.seed(0)
        personal_best = particle.personal_best[np.random.choice(len(particle.personal_best))]

        # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
        #                   self.params.position_dim,
        #                   dtype=np.float64)
        # cuda.memcpy_dtoh(teste8, self.position_g)
        # teste8.shape = 261, 10
        # l = []
        # for i in range(self.params.memory_size):
        #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
        #     l[-1] = max(l[-1])
        # self.l61.append(l)

        self.whatPersonal[particle_index] = (
            np.where(np.array(particle.personal_best) == personal_best))[0][0]

        # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
        #                   self.params.position_dim,
        #                   dtype=np.float64)
        # cuda.memcpy_dtoh(teste8, self.position_g)
        # teste8.shape = 261, 10
        # l = []
        # for i in range(self.params.memory_size):
        #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
        #     l[-1] = max(l[-1])
        # self.l62.append(l)

        if self.params.Xr_pool_type == 0:  # Apenas Populacao
            for p in self.population:
                if not personal_best == p or not particle == p:
                    if not particle >> p:
                        Xr_pool.append(p)
        elif self.params.Xr_pool_type == 1:  # Apenas Memoria
            for m in self.memory:
                if not personal_best == m or not particle == m:
                    if not particle >> m:
                        Xr_pool.append(m)
        elif self.params.Xr_pool_type == 2:  # Combinacao Memoria testeTempo250624 Populacao
            for m in self.memory:
                if not personal_best == m and not particle == m:
                    if not particle >> m:
                        Xr_pool.append(m)
            for p in self.population:
                if not personal_best == p and not particle == p and p not in Xr_pool and p.rank > particle.rank:
                    if not particle >> p:
                        Xr_pool.append(p)

        if self.params.DE_mutation_type == 0 and len(Xr_pool) >= 3:  # DE\rand\1\Bin
            # np.random.seed(0)
            Xr_list = np.random.choice(Xr_pool, 3, replace=False)

            if self.params.Xr_pool_type == 1:
                self.xr_list[particle_index * 5 + 0] = np.where(np.array(self.memory) == Xr_list[0])[0][0]
                self.xr_list[particle_index * 5 + 1] = np.where(np.array(self.memory) == Xr_list[1])[0][0]
                self.xr_list[particle_index * 5 + 2] = np.where(np.array(self.memory) == Xr_list[2])[0][0]
                self.xr_list[particle_index * 5 + 3] = -1
                self.xr_list[particle_index * 5 + 4] = -1

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l63.append(l)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)
            Xr3 = np.asarray(Xr_list[2].position)

            Xst = Xr1 + self.weights[5][particle_index] * (Xr2 - Xr3)
            # if particle_index == 0:
            #     print(Xst)
            # if particle_index == 127:
            #     print(Xst)
            Xst = Xst.tolist()
            # if particle_index == 0:
            #     print(Xst)
            # if particle_index == 127:
            #     print(Xst)
            Xst = self.check_position_limits(Xst)
            # if particle_index == 0:
            #     print(Xst)
            # if particle_index == 127:
            #     print(Xst)

            # np.random.seed(0)
            mutation_index = np.random.choice(self.params.position_dim)

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l64.append(l)

            self.mutation_index[particle_index] = mutation_index

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l65.append(l)

            # if particle_index < 5 or particle_index >= 123:
            #     print(mutation_index)

            # np.random.seed(0)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l66.append(l)

            mutation_chance = mutation_chance.astype(np.float64)

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l67.append(l)

            self.mutation_chance[particle_index * self.params.position_dim:(particle_index + 1)
                                                                           * self.params.position_dim] = mutation_chance.copy()

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l68.append(l)

            # if self.gpu:
            #     self.xst[particle_index*self.params.position_dim:(particle_index+1)*self.params.position_dim] = Xst

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l69.append(l)

            # self.teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                        self.params.objectives_dim,
            #                        dtype=np.float64)
            # self.teste8 = self.xst_fitness.copy()
            # self.teste8.shape = 261, 2

            for i in range(self.params.position_dim):
                if mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index:
                    Xst[i] = personal_best.position[i]
            if self.gpu:
                self.xst[
                particle_index * self.params.position_dim:(particle_index + 1) * self.params.position_dim] = Xst

            # teste8 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste8, self.position_g)
            # teste8.shape = 261, 10
            # l = []
            # for i in range(self.params.memory_size):
            #     l.append(abs(np.array(self.memory[i].position) - teste8[256 + i]))
            #     l[-1] = max(l[-1])
            # self.l610.append(l)

        elif self.params.DE_mutation_type == 1 and len(Xr_pool) >= 5:  # DE\rand\2\Bin
            Xr_list = np.random.choice(Xr_pool, 5, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)
            Xr3 = np.asarray(Xr_list[2].position)
            Xr4 = np.asarray(Xr_list[3].position)
            Xr5 = np.asarray(Xr_list[4].position)

            Xst = Xr1 + self.weights[5][particle_index] * ((Xr2 - Xr3) + (Xr4 - Xr5))
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            for i in range(self.params.position_dim):
                if (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = personal_best.position[i]

        elif self.params.DE_mutation_type == 2 and len(Xr_pool) >= 2:  # DE/Best/1/Bin
            Xr_list = np.random.choice(Xr_pool, 2, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)

            Xst = particle.global_best.position + self.weights[5][particle_index] * (Xr1 - Xr2)
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            for i in range(self.params.position_dim):
                if not (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = particle.global_best.position[i]

        elif self.params.DE_mutation_type == 3 and len(Xr_pool) >= 2:  # DE/Current-to-best/1/Bin
            Xr_list = np.random.choice(Xr_pool, 2, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)

            Xst = np.asarray(personal_best.position) + self.weights[5][particle_index] * (
                        (Xr1 - Xr2) + (np.asarray(particle.global_best.position) - np.asarray(personal_best.position)))
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            for i in range(self.params.position_dim):
                if not (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = particle.global_best.position[i]

        elif self.params.DE_mutation_type == 4 and len(Xr_pool) >= 3:  # DE/Current-to-rand/1/Bin
            Xr_list = np.random.choice(Xr_pool, 3, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)
            Xr3 = np.asarray(Xr_list[2].position)

            Xst = np.asarray(personal_best.position) + self.weights[5][particle_index] * (
                        (Xr1 - Xr2) + (Xr3 - np.asarray(personal_best.position)))
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            for i in range(self.params.position_dim):
                if not (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = particle.global_best.position[i]

        else:
            return

        if self.params.secondary_params:
            fit_eval = self.fitness_evaluation(self.fitness_function, Xst)
            Xst_fit = fit_eval[0]
        else:
            Xst_fit = self.fitness_evaluation(self.fitness_function, Xst)
            if self.gpu:
                # self.xst_fitness[particle_index*self.params.objectives_dim:(particle_index+1)*self.params.objectives_dim]\
                #     = Xst_fit.copy()
                self.xst_fitness[
                particle_index * self.params.objectives_dim:(particle_index + 1) * self.params.objectives_dim] \
                    = Xst_fit
        Xst_particle = Particle(self.params.position_min_value, self.params.position_max_value,
                                self.params.position_dim,
                                self.params.velocity_min_value, self.params.velocity_max_value,
                                self.params.objectives_dim, self.params.otimizations_type,
                                self.params.secondary_params)
        Xst_particle.fitness = Xst_fit
        Xst_particle.position = Xst
        if self.params.secondary_params:
            Xst_particle.secondary_params = fit_eval[1:]

        # self.teste = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
        #                  self.params.position_dim,
        #                  dtype=np.float64)
        # cuda.memcpy_dtoh(self.teste, self.xst_g)
        # self.teste.shape = 261, 10
        #
        # self.teste2 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
        #                   self.params.objectives_dim,
        #                   dtype=np.float64)
        # cuda.memcpy_dtoh(self.teste2, self.xst_fitness_g)
        # self.teste2.shape = 261, 2
        #
        # l = []
        # for i in range(len(self.teste2)):
        #     l.append(abs(np.array(self.xst_fitness[i] - self.teste2[i])))
        #     l[-1] = max(l[-1])
        # temp = np.where(np.array(l) > 1e-5)[0]
        # print(temp)
        # l2 = []
        # for i in range(len(self.teste)):
        #     l2.append(abs(np.array(self.xst[i] - self.teste[i])))
        #     l2[-1] = max(l2[-1])
        # temp2 = np.where(np.array(l2) > 1e-5)[0]
        # print(temp2)
        # pass

        # self.teste10.append([])
        # self.teste10[-1].append(self.xst)
        # self.teste10[-1].append(self.xst_fitness)
        # self.teste10[-1].append(self.xst_dominate)
        # self.teste10[-1].append(self.xst)

        # if particle_index == 80:
        #     pass

        # pq apenas atualiza a posicao testeTempo250624 a avaliação?
        if Xst_particle >> particle:
            # if Xst_particle >> particle and particle_index < 84:
            self.xst_dominate[particle_index] = 1
            # self.teste10[-1].append(self.xst)
            # self.teste10[-1].append(self.xst_fitness)
            # self.teste10[-1].append(self.xst_dominate)

            particle.fitness = Xst_fit
            particle.position = Xst

            # self.teste10[-1].append(self.xst)
            # self.teste10[-1].append(self.xst_fitness)
            # self.teste10[-1].append(self.xst_dominate)

            self.update_from_differential_mutation = True
            self.update_personal_best(particle)
            pass
        # else:
        #     self.teste10.append([])

    def memory_update(self):
        new_memory_candidates = []
        # l = []
        for f in self.fronts[0]:
            new_memory_candidates.append(f)
            # l.append(int(np.where(np.array(self.population) == f)[0]))
        for m in self.memory:
            if m not in new_memory_candidates:
                new_memory_candidates.append(m)
                # l.append(int(np.where(np.array(self.memory) == m)[0])+256)
        new_memory_front = self.fast_nondominated_sort(True, False, new_memory_candidates)

        # teste
        # l2 = []
        # for f in new_memory_front:
        #     if len(np.where(np.array(self.population) == f)[0]) > 0:
        #         l2.append(int(np.where(np.array(self.population) == f)[0]))
        #     else:
        #         l2.append(int(np.where(np.array(self.memory) == f)[0]) + 256)
        # print(l2)

        new_memory = []
        if len(new_memory_front) <= self.params.memory_size:
            for f in new_memory_front:
                new_memory.append(f)
        else:
            self.crowding_distance(new_memory_front)
            new_memory_front.sort(key=lambda x: x.crowd_distance)
            i = len(new_memory_front) - 1
            while len(new_memory) < self.params.memory_size:
                new_memory.append(new_memory_front[i])
                i = i - 1

        # self.memory2 = self.memory
        # self.memory3 = l
        self.memory = copy.deepcopy(new_memory)

    # def update_personal_best_gpu(self, particle):
    def update_personal_best_gpu(self):
        # i = len(particle.personal_best)

        # teste = np.zeros(self.params.population_size*2+self.params.memory_size, dtype= np.int32)
        # cuda.memcpy_dtoh(teste, self.personal_best_g)
        # print(teste)

        update_personal_best = self.mod.get_function("update_personal_best")
        update_personal_best(self.personal_best_position_g, self.personal_best_velocity_g, self.personal_best_fitness_g,
                             self.params.objectives_dim_g, self.params.position_dim_g, self.position_g, self.fitness_g,
                             self.params.personal_guide_array_size_g,
                             block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
        cuda.Context.synchronize()

        # teste = np.zeros((self.params.population_size*2+self.params.memory_size)*self.params.position_dim*
        #                  self.params.personal_guide_array_size, dtype=np.float64)
        # cuda.memcpy_dtoh(teste, self.personal_best_position_g)
        # teste2 = np.zeros((self.params.population_size * 2 + self.params.memory_size) * self.params.objectives_dim *
        #                  self.params.personal_guide_array_size, dtype=np.float64)
        # cuda.memcpy_dtoh(teste2, self.personal_best_fitness_g)
        # pass

        # if particle.personal_best[0] is None:
        #     pass
            # new_personal_best = Particle(self.params.position_min_value,self.params.position_max_value, self.params.position_dim,
            #                     self.params.velocity_min_value, self.params.velocity_max_value,
            #                     self.params.objectives_dim,self.params.otimizations_type,
            #                     self.params.secondary_params)
            # new_personal_best.position = particle.position
            # new_personal_best.fitness = particle.fitness
            # particle.personal_best = []
            # particle.personal_best.append(new_personal_best)

        # else:
        #     pass
            # remove_list = []
            # include_flag = False
            # for s in particle.personal_best:
            #     if particle == s:
            #         break
            #     if particle >> s:
            #         include_flag = True
            #         if s not in remove_list:
            #             remove_list.append(s)
            #     if not particle << s:
            #         i = i - 1
            # if len(remove_list) > 0:
            #     for r in remove_list:
            #         particle.personal_best.remove(r)
            # # tem algo estranho aqui. Se i = 1 testeTempo250624 include_flag = true, significa que a particula
            # # é dominada por 1 particula do conjunto testeTempo250624 domina pelo menos 1 outra
            # # tentar obter um exemplo. talvez um ou exclusivo?
            # if i == 0 or include_flag:
            #     new_personal_best = Particle(self.params.position_min_value,self.params.position_max_value, self.params.position_dim,
            #                     self.params.velocity_min_value, self.params.velocity_max_value,
            #                     self.params.objectives_dim,self.params.otimizations_type,
            #                     self.params.secondary_params)
            #     new_personal_best.position = particle.position
            #     new_personal_best.fitness = particle.fitness
            #
            #     if self.params.personal_guide_array_size > 0 and len(particle.personal_best) == self.params.personal_guide_array_size:
            #         particle.personal_best.pop(0)
            #     particle.personal_best.append(new_personal_best)

    def update_personal_best(self, particle):
        i = len(particle.personal_best)
        if particle.personal_best[0] is None:
            new_personal_best = Particle(self.params.position_min_value, self.params.position_max_value,
                                         self.params.position_dim,
                                         self.params.velocity_min_value, self.params.velocity_max_value,
                                         self.params.objectives_dim, self.params.otimizations_type,
                                         self.params.secondary_params)

            # esse trecho faz a position testeTempo250624 fitness como o emsmo objeto da particula no eprsonal best
            # testeTempo250624 na particula atual. Quando a particula troca sua position testeTempo250624 fitness, seus valores sao
            # tb alterados. Acho que deveria guardar uam copai deles, ja q eles deveriam guardar aos melhores
            # nesse momento testeTempo250624 nao ao longo dos passos comoe stava ate agora

            # new_personal_best.position = particle.position
            # new_personal_best.fitness = particle.fitness
            new_personal_best.position = particle.position.copy()
            new_personal_best.fitness = particle.fitness.copy()

            particle.personal_best = []
            particle.personal_best.append(new_personal_best)
        else:
            remove_list = []
            include_flag = False
            for s in particle.personal_best:
                if particle == s:
                    break
                if particle >> s:
                    include_flag = True
                    if s not in remove_list:
                        remove_list.append(s)
                if not particle << s:
                    i = i - 1
            if len(remove_list) > 0:
                for r in remove_list:
                    particle.personal_best.remove(r)
            # tem algo estranho aqui. Se i = 1 testeTempo250624 include_flag = true, significa que a particula
            # é dominada por 1 particula do conjunto testeTempo250624 domina pelo menos 1 outra
            # tentar obter um exemplo. talvez um ou exclusivo?
            if i == 0 or include_flag:
                new_personal_best = Particle(self.params.position_min_value, self.params.position_max_value,
                                             self.params.position_dim,
                                             self.params.velocity_min_value, self.params.velocity_max_value,
                                             self.params.objectives_dim, self.params.otimizations_type,
                                             self.params.secondary_params)

                # esse trecho faz a position testeTempo250624 fitness como o emsmo objeto da particula no eprsonal best
                # testeTempo250624 na particula atual. Quando a particula troca sua position testeTempo250624 fitness, seus valores sao
                # tb alterados. Acho que deveria guardar uam copai deles, ja q eles deveriam guardar aos melhores
                # nesse momento testeTempo250624 nao ao longo dos passos comoe stava ate agora

                # new_personal_best.position = particle.position
                # new_personal_best.fitness = particle.fitness
                new_personal_best.position = particle.position.copy()
                new_personal_best.fitness = particle.fitness.copy()

                if self.params.personal_guide_array_size > 0 and len(
                        particle.personal_best) == self.params.personal_guide_array_size:
                    particle.personal_best.pop(0)
                particle.personal_best.append(new_personal_best)

    def global_best_attribution_gpu(self, use_copy_population=False):
        fitness = []

        for m in self.memory:
            fitness.append(m.fitness)
        for p in self.population:
            fitness.append(p.fitness)
        if use_copy_population:
            for c in self.population_copy:
                fitness.append(c.fitness)

        fitness = np.array(fitness, dtype=np.float64)
        sigma = np.zeros((len(fitness), 3), dtype=np.float64)
        # fitness = np.array(fitness, dtype=np.float64)
        # sigma = np.zeros((len(fitness), 3), dtype=np.float64)

        fitness_g = cuda.mem_alloc(fitness.nbytes)
        dim_g = cuda.mem_alloc(np.array(fitness.shape, dtype=np.int32).nbytes)
        sigma_g = cuda.mem_alloc(sigma.nbytes)

        cuda.memcpy_htod(fitness_g, fitness)
        cuda.memcpy_htod(dim_g, np.array(fitness.shape, dtype=np.int32))
        cuda.memcpy_htod(sigma_g, sigma)

        f = open('testes/ordenacao.cu')
        code = f.read()
        f.close()
        mod = SourceModule(code)

        sigma_eval = mod.get_function("sigma_eval")

        block_x = len(fitness)
        sigma_eval(fitness_g, dim_g, sigma_g, block=(block_x, 1, 1), grid=(1, 1, 1))
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(sigma, sigma_g)
        i = 0
        for m in self.memory:
            m.sigma_value = sigma[i, :]
            i += 1
        for p in self.population:
            p.sigma_value = sigma[i, :]
            i += 1
        if use_copy_population:
            for c in self.population_copy:
                c.sigma_value = sigma[i, :]
                i += 1

        fronts = []
        tam = []
        pos_mem = []

        if use_copy_population:
            pool = np.zeros(2 * len(self.population), dtype=np.int32)
            nearest = np.zeros(2 * len(self.population), dtype=np.int32)
            pos_mem = np.zeros(2 * len(self.population), dtype=np.int32)
        else:
            pool = np.zeros(len(self.population), dtype=np.int32)
            nearest = np.zeros(len(self.population), dtype=np.int32)
            pos_mem = np.zeros(len(self.population), dtype=np.int32)

        for i in range(len(self.fronts)):
            for j in range(len(self.fronts[i])):
                fronts.append(np.where(np.array(self.population) == self.fronts[i][j])[0][0])
            tam.append(len(self.fronts[i]))
        fronts = np.array(fronts, dtype=np.int32)
        tam = np.array(tam, dtype=np.int32)

        for i in range(len(self.population)):
            if self.population[i].rank == 0:
                pool[i] = -1
                if len(np.where(np.array(self.memory) == self.population[i])[0]) == 1:
                    pos_mem[i] = np.where(np.array(self.memory) == self.population[i])[0][0]
                else:
                    pos_mem[i] = -1

            else:
                pool[i] = self.population[i].rank - 1
                pos_mem[i] = -2
        if use_copy_population:
            for i in range(len(self.population_copy)):
                if self.population_copy[i].rank == 0:
                    pool[i + len(self.population)] = -1
                    if len(np.where(np.array(self.memory) == self.population_copy[i])[0]) == 1:
                        pos_mem[i + len(self.population)] = \
                        np.where(np.array(self.memory) == self.population_copy[i])[0][0]
                    else:
                        pos_mem[i + len(self.population)] = -1
                else:
                    pool[i + len(self.population)] = self.population[i].rank - 1
                    pos_mem[i + len(self.population)] = -2

        fronts_g = cuda.mem_alloc(fronts.nbytes)
        tam_g = cuda.mem_alloc(tam.nbytes)
        tam_memoria_g = cuda.mem_alloc(np.array([1], dtype=np.int32).nbytes)
        pool_g = cuda.mem_alloc(pool.nbytes)
        nearest_g = cuda.mem_alloc(nearest.nbytes)
        pos_mem_g = cuda.mem_alloc(pos_mem.nbytes)

        cuda.memcpy_htod(fronts_g, fronts)
        cuda.memcpy_htod(tam_g, tam)
        cuda.memcpy_htod(tam_memoria_g, np.array([len(self.memory)], np.int32))
        cuda.memcpy_htod(pool_g, pool)
        cuda.memcpy_htod(nearest_g, nearest)
        cuda.memcpy_htod(pos_mem_g, pos_mem)

        f = open('testes/ordenacao.cu')
        code = f.read()
        f.close()
        mod = SourceModule(code)

        sigma_nearest = mod.get_function("sigma_nearest")
        block_x = len(self.population) + len(self.population_copy)
        sigma_nearest(fronts_g, tam_g, tam_memoria_g, pool_g, sigma_g, nearest_g, pos_mem_g,
                      block=(block_x, 1, 1), grid=(1, 1, 1))
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(nearest, nearest_g)

        for i in range(len(self.population)):
            if pool[i] != -1:
                self.population[i].global_best = self.population[nearest[i]]
                # self.population[i].global_best2 = self.population[nearest[i]]
            else:
                self.population[i].global_best = self.memory[nearest[i]]
                # self.population[i].global_best2 = self.memory[nearest[i]]
        if use_copy_population:
            for i in range(len(self.population_copy)):
                if pool[i] != -1:
                    self.population_copy[i].global_best = (
                        self.population)[nearest[i + len(self.population)]]
                    # self.population_copy[i].global_best2 = (
                    #     self.population)[nearest[i + len(self.population)]]
                else:
                    self.population_copy[i].global_best = (
                        self.memory)[nearest[i + len(self.population)]]
                    # self.population_copy[i].global_best2 = (
                    #     self.memory)[nearest[i + len(self.population)]]

        # validacao global_best
        # l = []
        # for i in self.population:
        #     l.append(i.global_best == i.global_best2)
        # print('soma', sum(l), l)
        # l2 = []
        # for i in self.population_copy:
        #     l2.append(i.global_best == i.global_best2)
        # print('soma', sum(l2), l2)
        # a = 1

    def global_best_attribution(self, use_copy_population=False):
        if self.params.global_best_attribution_type == 0 or self.params.global_best_attribution_type == 1:
            for m in self.memory:
                self.sigma_eval(m)
            # Sigma com memoria apenas.
            if self.params.global_best_attribution_type == 0:
                for p in self.population:
                    self.sigma_eval(p)
                    self.sigma_nearest(p, self.memory)
                if use_copy_population:
                    for c in self.population_copy:
                        self.sigma_eval(c)
                        self.sigma_nearest(c, self.memory)
            # Sigma por fronteiras.
            if self.params.global_best_attribution_type == 1:
                for p in self.population:
                    self.sigma_eval(p)
                if use_copy_population:
                    for c in self.population_copy:
                        self.sigma_eval(c)
                for p in self.population:
                    if p.rank == 0:
                        self.sigma_nearest(p, self.memory)
                    else:
                        self.sigma_nearest(p, self.fronts[p.rank - 1])
                for c in self.population_copy:
                    if c.rank == 0:
                        self.sigma_nearest(c, self.memory)
                    else:
                        self.sigma_nearest(c, self.fronts[c.rank - 1])
            # Random na memoria - mas apenas entra no if se for 0 ou 1
            if self.params.global_best_attribution_type == 2:
                for p in self.population:
                    p.global_best = self.memory[np.random.choice(len(self.memory))]
                if use_copy_population:
                    for c in self.population_copy:
                        c.global_best = self.memory[np.random.choice(len(self.memory))]
            # Random por fronteiras
            if self.params.global_best_attribution_type == 3:
                for p in self.population:
                    if p.rank == 0:
                        p.global_best = self.memory[np.random.choice(len(self.memory))]
                    else:
                        p.global_best = self.fronts[p.rank - 1][np.random.choice(len(self.fronts[p.rank - 1]))]
                if use_copy_population:
                    for c in self.population:
                        if c.rank == 0:
                            c.global_best = self.memory[np.random.choice(len(self.memory))]
                        else:
                            c.global_best = self.fronts[c.rank - 1][np.random.choice(len(self.fronts[c.rank - 1]))]
            a = 1

    def check_stopping_criteria(self):
        # if self.params.max_fitness_eval != 0 and self.fitness_eval_count >= self.params.max_fitness_eval:
        #     self.stopping_criteria_reached = True
        # if self.params.max_iterations != 0 and self.generation_count == self.params.max_iterations:
        #     self.stopping_criteria_reached = True
        if self.generation_count == self.max_num_iters:
            self.stopping_criteria_reached = True

    def grafico_espaco_objetivo_2D(self):
        f1 = []
        f2 = []
        plt.clf()
        for i in self.population:
            f1.append(i.fitness[0])
            f2.append(i.fitness[1])

        p = get_problem('zdt1', ndim=2).pareto_front()
        plt.plot(f1, f2, 'bo', p[:, 0], p[:, 1], 'ro')
        # fig.clear()
        # axis.plot(f1, f2)
        plt.savefig('testesSimulacao/' + str(self.generation_count) + '.png')

    def run(self, func='ZDT1', teste_grafico=False, teste_tempo = False):

        #teste de tempo
        cpu = 10*[0]
        gpu = 10*[0]
        teste = {}

        with tqdm(total=self.params.max_fitness_eval, leave=False) as pbar:

            # Inicia populacao - cria cada particula inicial com posições aleatorias
            # start = dt.now()
            # total = dt.now() - start

            # np.random.seed(0)
            # print(np.random.get_state())

            # f = open('results_zdt1.pkl', 'rb')
            # results = pickle.load(f)
            # f.close()
            # f = open('results_zdt1.pkl', 'wb')
            # results['seed'].append(np.random.get_state())
            # pickle.dump(results, f)
            # f.close()

            # f = open('testesSimulacao/results_test.pkl', 'rb')
            # results = pickle.load(f)
            # f.close()
            # np.random.set_state(results['seed'][4])
            self.init_population()
            # print(self.population[0].position)

            prev_fitness_eval = 0

            ## avalia fitness testeTempo250624 sigma da populacao
            # for p in self.population:
            #     if self.params.secondary_params:
            #         fit_eval = self.fitness_evaluation(self.fitness_function, p.position)
            #         p.fitness = fit_eval[0]
            #         p.secondary_params = fit_eval[1:]
            #     else:
            #         p.fitness = self.fitness_evaluation(self.fitness_function, p.position)
            #     self.update_personal_best(p)  # inicialmente cada particula é seu personal best

            if self.gpu:
                # if func == 'ZDT1':
                # zdt1 = self.mod.get_function("zdt1")
                function = self.mod.get_function("function")
                # zdt1(self.position_g, self.params.position_dim_g, self.fitness_g,
                #      block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
                function(self.params.func_n_g, self.position_g, self.params.position_dim_g,
                         self.fitness_g, self.alpha_g,
                     block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
                cuda.Context.synchronize()
                self.fitness_eval_count += self.params.population_size
                self.update_personal_best_gpu()

                ##teste
                # teste_f = np.zeros(128*3*2, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 128*3, 2
                # pass
                # teste_p = np.zeros(128*3 * 10, np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 128*3, 10
                # pass
                # teste_v = np.zeros(256*3 * 12, np.float64)
                # cuda.memcpy_dtoh(teste_v, self.velocity_g)
                # teste_v.shape = 256*3, 12
                # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                # plt.show()
                # pass

            # teste = np.zeros(128 * 30, dtype=np.float64)
            # cuda.memcpy_dtoh(teste, self.personal_best_position_g)
            # teste.shape = 128, 3, 10
            # teste2 = np.zeros(128 * 6, dtype=np.float64)
            # cuda.memcpy_dtoh(teste2, self.personal_best_fitness_g)
            # teste2.shape = 128, 3, 2
            # pass

            # self.test = np.zeros(len(self.population)*3, dtype=np.float64)
            # cuda.memcpy_dtoh(self.test, self.fitness_g)

            # encontra fronteiras das populacao
            # start1 = dt.now()

            # teste2 = np.zeros(128, dtype=np.int32)
            # cuda.memcpy_dtoh(teste2, self.fronts_g)
            # teste = np.zeros(261 * 30, dtype=np.float64)
            # cuda.memcpy_dtoh(teste, self.aux3_g)
            # teste.shape = 261, 3, 10
            # teste3 = np.zeros(261 * 30, dtype=np.float64)
            # cuda.memcpy_dtoh(teste3, self.personal_best_position_g)
            # teste3.shape = 261, 3, 10

            # self.fronts = self.fast_nondominated_sort()

            if self.gpu:
                div = int(self.params.population_size/16)
                fast_nondominated_sort = self.mod.get_function("fast_nondominated_sort")
                fast_nondominated_sort(self.fitness_g, self.params.objectives_dim_g,
                                       self.domination_counter_g, self.params.population_size_g,
                                       self.params.otimizations_type_g, self.params.objectives_dim_g,
                                       block=(16, 32, 1),
                                       grid=(div, int(div/2), 1))
                cuda.Context.synchronize()
                # teste
                # teste_dc = np.zeros(385 * 385, np.int32)
                # cuda.memcpy_dtoh(teste_dc, self.domination_counter_g)
                # teste_dc.shape = 385, 385
                # teste_f = np.zeros(384*3, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 384, 3
                # pass
                # teste_p = np.zeros(384 * 10, np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 384, 10
                # pass

                fast_nondominated_sort2 = self.mod.get_function("fast_nondominated_sort2")
                fast_nondominated_sort2(self.domination_counter_g, self.params.population_size_g,
                                        self.params.population_size_g,
                                        block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
                cuda.Context.synchronize()

                #teste
                # teste_dc2 = np.zeros(129 * 128, np.int32)
                # cuda.memcpy_dtoh(teste_dc2, self.domination_counter_g)
                # teste_dc2.shape = 129, 128

                fast_nondominated_sort3 = self.mod.get_function("fast_nondominated_sort3")
                fast_nondominated_sort3(self.domination_counter_g, self.params.population_size_g,
                                        self.params.population_size_g, self.fronts_g, self.tams_fronts_g,
                                        self.rank_g,
                                        block=(1, 1, 1), grid=(1, 1, 1))
                cuda.Context.synchronize()

                #teste
                # teste_fr = np.zeros(2*self.params.population_size, dtype=np.int32)
                # cuda.memcpy_dtoh(teste_fr, self.fronts_g)
                # teste_tam = np.zeros(256, dtype=np.int32)
                # cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                # teste_f = np.zeros(384*2, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 384, 2
                # pass

            # print('')
            # print(teste)
            # print(teste2)
            # i = 1
            # print(teste[0:teste2[0]])
            # while teste2[i] != -1:
            #     print(teste[sum(teste2[0:i]):sum(teste2[0:i+1])])
            #     i += 1
            # for i in range(self.params.population_size):
            #     print((i, self.population[i].rank))
            # print("")
            # pass

            # cpu = False
            # if cpu:
            # self.fronts = self.fast_nondominated_sort()
            # else:
            # fronts_2 = self.fast_nondominated_sort_gpu()
            # start2 = dt.now()
            # self.fronts = self.fast_nondominated_sort_gpu()
            # print('gpu', dt.now()-start2)

            # atualiza memoria
            # if len(self.fronts[0]) <= self.params.memory_size:
            #     for f in self.fronts[0]:
            #         self.memory.append(f)
            # else:
            #     self.crowding_distance(self.fronts[0])
            #     self.fronts[0].sort(key=lambda x: x.crowd_distance)
            #     j = len(self.fronts[0]) - 1
            #     while len(self.memory) < self.params.memory_size:
            #         new_particle = Particle(self.params.position_min_value, self.params.position_max_value,
            #                                 self.params.position_dim,
            #                                 self.params.velocity_min_value, self.params.velocity_max_value,
            #                                 self.params.objectives_dim, self.params.otimizations_type,
            #                                 self.params.secondary_params, self.gpu)
            #         new_particle.position = self.fronts[0][j].position.copy()
            #         new_particle.fitness = self.fronts[0][j].fitness.copy()
            #         self.memory.append(new_particle)  # modificacao a ser estudada depois
            #         j = j - 1

            if self.gpu:

                # # teste
                # teste_fr = np.zeros(2 * self.params.population_size, dtype=np.int32)
                # cuda.memcpy_dtoh(teste_fr, self.fronts_g)
                # teste_f = np.zeros(384 * 2, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 384, 2
                # pass

                tam_fronts = np.zeros(2*self.params.population_size, dtype=np.int32)
                cuda.memcpy_dtoh(tam_fronts, self.tams_fronts_g)
                # atualiza memoria pela GPU
                if tam_fronts[0] <= self.params.memory_size:

                    cuda.memcpy_htod(self.params.current_memory_size_g, tam_fronts[0])

                    memory_inicialization1 = self.mod.get_function("memory_inicialization1")
                    memory_inicialization1(self.position_g, self.fitness_g, self.fronts_g,
                                           self.params.position_dim_g, self.params.objectives_dim_g,
                                           self.params.population_size_g,
                                           block=(int(tam_fronts[0]), 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # tam = 128
                    # teste_f = np.zeros(tam*3*2, np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = tam*3, 2
                    # pass
                    # teste_p = np.zeros(tam*3*10, np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = tam*3, 10
                    # pass

                else:
                    cuda.memcpy_htod(self.params.current_memory_size_g, self.params.memory_size)
                    # teste2 = np.zeros(2 * self.params.population_size, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste2, self.crowding_distance_g)
                    # print(teste2)
                    crowding_distance_inicialization = self.mod.get_function("crowding_distance_inicialization")
                    crowding_distance_inicialization(self.crowding_distance_g,
                                                     block=(2 * self.params.population_size, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()
                    teste2 = np.zeros(2 * self.params.population_size, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste2, self.crowding_distance_g)
                    # print(teste2)
                    # teste3 = np.zeros(self.params.population_size, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste3, self.fronts_g)
                    i_g = cuda.mem_alloc(np.array([1], np.int32).nbytes)
                    for i in range(self.params.objectives_dim):
                        # print(teste3[0:teste[0]])
                        cuda.memcpy_htod(i_g, np.array([i], dtype=np.int32))
                        front_sort = self.mod.get_function("front_sort")
                        front_sort(self.fronts_g, self.tams_fronts_g, self.fitness_g,
                                   self.params.objectives_dim_g, i_g, self.crowding_distance_g,
                                   block=(1, 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()
                        # teste3 = np.zeros(self.params.population_size, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste3, self.fronts_g)
                        # print(teste3[0:teste[0]])
                        crowding_distance = self.mod.get_function("crowding_distance")
                        crowding_distance(self.fronts_g, self.tams_fronts_g, self.fitness_g,
                                          self.params.objectives_dim_g, self.tams_fronts_g, i_g,
                                          self.crowding_distance_g,
                                          block=(int(teste[0] - 2), 1, 1), grid=(1, 1, 1))
                        # teste3 = np.zeros(self.params.population_size*2, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste3, self.crowding_distance_g)
                        # print(teste3[:teste[0]+2])
                        # pass
                    # self.crowding_distance(self.fronts[0])
                    # self.fronts[0].sort(key=lambda x: x.crowd_distance)
                    front_sort_crowding_distance = self.mod.get_function("front_sort_crowding_distance")
                    front_sort_crowding_distance(self.fronts_g, self.tams_fronts_g,
                                                 self.crowding_distance_g,
                                                 block=(1, 1, 1), grid=(1, 1, 1))
                    # teste3 = np.zeros(self.params.population_size * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste3, self.crowding_distance_g)
                    # print(teste3[:teste[0] + 2])
                    # teste4 = np.zeros(self.params.population_size, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste4, self.fronts_g)
                    # print(teste4[:teste[0] + 2])
                    # j = len(self.fronts[0])-1
                    # while len(self.memory) < self.params.memory_size:
                    #     self.memory.append(self.fronts[0][j])
                    #     j = j - 1

                    # if self.generation_count == 0:
                    #     initial_memory_velocity1 = self.mod.get_function("initial_memory_velocity1")
                    #     initial_memory_velocity1(self.initial_memory_g, self.fronts_g,
                    #                              block=(self.params.population_size, 1, 1),
                    #                              grid=(1, 1, 1))
                    #     cuda.Context.synchronize()
                    #
                    #     # teste3 = np.zeros(261*10, dtype=np.float64)
                    #     # cuda.memcpy_dtoh(teste3, self.velocity_g)
                    #     # teste3.shape = 261, 10
                    #     teste3 = np.zeros(128, dtype=np.int32)
                    #     cuda.memcpy_dtoh(teste3, self.initial_memory_g)
                    #     teste4 = np.zeros(128, dtype=np.int32)
                    #     cuda.memcpy_dtoh(teste4, self.fronts_g)

                    memory_inicialization2 = self.mod.get_function("memory_inicialization2")
                    memory_inicialization2(self.position_g, self.fitness_g, self.fronts_g,
                                           self.params.position_dim_g, self.params.objectives_dim_g,
                                           self.params.population_size_g,
                                           block=(self.params.memory_size, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                # #teste
                # teste_f = np.zeros(384*2, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 384, 2
                # teste_p = np.zeros(384 * 10, np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 384, 10
                # teste_v = np.zeros(384 * 10, np.float64)
                # cuda.memcpy_dtoh(teste_v, self.velocity_g)
                # teste_v.shape = 384, 10
                # teste_cur = np.zeros(1, np.int32)
                # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                # # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                # # plt.show()
                # pass

                # trecho usado apenas para igualar as duas simulações, apagar depois
                # teste = np.zeros(self.params.position_dim * (2 * self.params.population_size + self.params.memory_size),
                #                  dtype=np.float64)
                # cuda.memcpy_dtoh(teste, self.position_g)
                # teste.shape = 2 * self.params.population_size + self.params.memory_size, self.params.position_dim
                # teste2 = np.zeros(
                #     self.params.objectives_dim * (2 * self.params.population_size + self.params.memory_size),
                #     dtype=np.float64)
                # cuda.memcpy_dtoh(teste2, self.fitness_g)
                # teste2.shape = 2 * self.params.population_size + self.params.memory_size, self.params.objectives_dim
                # if abs(self.memory[0].position[0] - teste[2 * self.params.population_size][0]) > 1e-3:
                #     print("\ntroca")
                #     temp = teste[2 * self.params.population_size].copy()
                #     teste[2 * self.params.population_size] = teste[2 * self.params.population_size + 1].copy()
                #     teste[2 * self.params.population_size + 1] = temp.copy()
                #     temp = teste2[2 * self.params.population_size].copy()
                #     teste2[2 * self.params.population_size] = teste2[2 * self.params.population_size + 1].copy()
                #     teste2[2 * self.params.population_size + 1] = temp.copy()
                # cuda.memcpy_htod(self.position_g, teste.flatten())
                # cuda.memcpy_htod(self.fitness_g, teste2.flatten())

                # l4 = []
                # for i in range(len(self.memory)):
                #     l4.append(abs(np.array(self.memory[i].position) - teste[256 + i]))
                #     l4[-1] = max(l4[-1])
                # pass
                # print(teste3[:teste[0] + 2])
                # teste4 = np.zeros(self.params.objectives_dim *
                #                   (2*self.params.population_size + self.params.memory_size), dtype=np.float64)
                # cuda.memcpy_dtoh(teste4, self.fitness_g)
                # teste5 = np.zeros(self.params.population_size, dtype=np.int32)
                # cuda.memcpy_dtoh(teste5, self.fronts_g)
                # pass
                # teste3 = np.zeros((self.params.population_size*2+self.params.memory_size)
                #                   *self.params.position_dim, dtype=np.float64)
                # cuda.memcpy_dtoh(teste3, self.position_g)
                # teste3.shape = 3*128, 10
                # teste4 = np.zeros((self.params.population_size * 2 + self.params.memory_size)
                #                   * self.params.objectives_dim, dtype=np.float64)
                # cuda.memcpy_dtoh(teste4, self.fitness_g)
                # teste4.shape = 3 * 128, 2
                # print(teste4)
                # pass

            # teste6 = np.zeros((self.params.population_size * 2 + self.params.memory_size) *
            #                   self.params.position_dim,
            #                   dtype=np.float64)
            # cuda.memcpy_dtoh(teste6, self.position_g)
            # teste6.shape = 261, 10

            # Main loop
            # teste_dict = {}

            # teste
            # teste_cur = np.zeros(1, np.int32)
            # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
            # teste_f = np.zeros(128*3*2, np.float64)
            # cuda.memcpy_dtoh(teste_f, self.fitness_g)
            # teste_f.shape = 128*3, 2
            # teste_p = np.zeros(128 * 3 * 10, np.float64)
            # cuda.memcpy_dtoh(teste_p, self.position_g)
            # teste_p.shape = 128 * 3, 10
            # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
            # plt.show()
            # pass

            while not self.stopping_criteria_reached:
                # print('iter', self.generation_count)
                # teste_dict[self.generation_count] = {}
                # self.grafico_espaco_objetivo_2D()

                # encontra os melhores globais de cada particula
                if 2 <= self.params.DE_mutation_type <= 4:  # Somente se for necessario na mutação do DE
                    self.global_best_attribution()  # 0:00:00.000053 de  0:00:58.232154

                # teste pb
                # print('\nteste antes pb differential mutation')
                # teste = np.zeros(384 * 30, dtype=np.float64)
                # cuda.memcpy_dtoh(teste, self.personal_best_position_g)
                # teste.shape = 384, 3, 10
                # teste2 = np.zeros(384 * 6, dtype=np.float64)
                # cuda.memcpy_dtoh(teste2, self.personal_best_fitness_g)
                # teste2.shape = 384, 3, 2
                # teste3 = np.zeros(384 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste3, self.position_g)
                # teste3.shape = 384, 10
                # teste4 = np.zeros(384 * 2, dtype=np.float64)
                # cuda.memcpy_dtoh(teste4, self.fitness_g)
                # teste4.shape = 384, 2
                # pass
                #
                # l = []
                # l3 = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(abs(np.array(self.population[i].personal_best[j].position) - teste[i][k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # num = []
                # for i in range(len(self.population)):
                #     num.append(0)
                #     for j in range(3):
                #         if teste[i][j][0] != 1e10:
                #             num[-1] += 1
                #     num[-1] -= len(self.population[i].personal_best)
                # print(np.where(np.array(num) != 0)[0])
                #
                # l = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(abs(np.array(self.population[i].personal_best[j].fitness) - teste2[i][k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # for i in range(len(self.population)):
                #     l2 = abs(np.array(self.population[i].position) - teste3[i])
                #     l.extend(l2)
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # for i in range(len(self.population)):
                #     l2 = abs(np.array(self.population[i].fitness) - teste4[i])
                #     l.extend(l2)
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                # pass

                for i, p in enumerate(self.population):  # 0:00:01.215242 de 0:00:45.768188
                    start = dt.now()
                    # self.differential_mutation(p, i)
                    cpu[0] += (dt.now() - start).total_seconds()

                    # teste
                    # if i == 80:
                    #     self.differential_mutation(p, i)
                    # else:
                    #     self.differential_mutation(p, i)

                # teste random
                # print(self.mutation_chance[:5])

                if self.gpu:
                    # print("wp cpu", self.whatPersonal[80])
                    # implementar a escolha aleatoria depois dentro da gpu. Por enquanto considerei sempre 0 dentro
                    # do kernel.
                    # self.whatPersonal = np.zeros(2*self.params.population_size+self.params.memory_size,
                    #                              dtype= np.int32)
                    # cuda.memcpy_htod(self.whatPersonal_g, self.whatPersonal)
                    # cuda.memcpy_htod(self.xr_list_g, self.xr_list)
                    # cuda.memcpy_htod(self.mutation_index_g, self.mutation_index)
                    # cuda.memcpy_htod(self.mutation_chance_g, self.mutation_chance)

                    # teste
                    # teste = np.zeros(261 * 5, np.int32)
                    # cuda.memcpy_dtoh(teste, self.xr_list_g)

                    start = dt.now()

                    # teste
                    # teste_f = np.zeros(128 * 3 * 2, np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 128 * 3, 2
                    # teste_p = np.zeros(128 * 3 * 10, np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = 128 * 3, 10
                    # pass

                    cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))
                    # if func == 'ZDT1':
                    differential_mutation = self.mod.get_function("differential_mutation")
                    differential_mutation(self.params.func_n_g, self.params.Xr_pool_type_g,
                                          self.params.population_size_g, self.params.memory_size_g,
                                          self.position_g, self.params.position_dim_g,
                                          self.personal_best_position_g,
                                          self.params.personal_guide_array_size_g,
                                          self.fitness_g, self.params.objectives_dim_g,
                                          self.params.otimizations_type_g, self.xr_pool_g,
                                          self.params.DE_mutation_type_g, self.xr_list_g,
                                          self.weights_g, self.xst_g,
                                          self.params.position_min_value_g, self.params.position_max_value_g,
                                          self.params.secondary_params_g,
                                          self.xst_fitness_g, self.xst_dominate_g, self.personal_best_fitness_g,
                                          self.personal_best_velocity_g, self.personal_best_tam_g,
                                          self.update_from_differential_mutation_g, self.seed_g, self.alpha_g,
                                          block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # teste
                    # teste_f = np.zeros(128 * 3 * 2, np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 128 * 3, 2
                    # teste_p = np.zeros(128 * 3 * 10, np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = 128 * 3, 10
                    # pass

                    gpu[0] += (dt.now() - start).total_seconds()

                    # teste
                    # teste_f = np.zeros(384 * 2, np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # teste_p = np.zeros(384 * 10, np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = 384, 10
                    # teste_v = np.zeros(384 * 10, np.float64)
                    # cuda.memcpy_dtoh(teste_v, self.velocity_g)
                    # teste_v.shape = 384, 10
                    # teste_cur = np.zeros(1, np.int32)
                    # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                    # # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                    # # plt.show()
                    # pass

                    # teste = np.zeros(261, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste, self.update_from_differential_mutation_g)
                    # pass

                    # teste pb
                    # print('\nteste depois pb differential mutation')
                    # teste = np.zeros(261 * 30, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste, self.personal_best_position_g)
                    # teste.shape = 261, 3, 10
                    # teste2 = np.zeros(261 * 6, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste2, self.personal_best_fitness_g)
                    # teste2.shape = 261, 3, 2
                    # teste3 = np.zeros(261 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste3, self.position_g)
                    # teste3.shape = 261, 10
                    # teste4 = np.zeros(261 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste4, self.fitness_g)
                    # teste4.shape = 261, 2
                    #
                    # l = []
                    # l3 = []
                    # for i in range(len(self.population)):
                    #     for j in range(len(self.population[i].personal_best)):
                    #         l2 = []
                    #         for k in range(3):
                    #             l2.append(
                    #                 list(abs(np.array(self.population[i].personal_best[j].position) - teste[i][k])))
                    #         l2.sort()
                    #         l.extend(l2[0])
                    #         l3.append((i, j))
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    #
                    # num = []
                    # for i in range(len(self.population)):
                    #     num.append(0)
                    #     for j in range(3):
                    #         if teste[i][j][0] != 1e10:
                    #             num[-1] += 1
                    #     num[-1] -= len(self.population[i].personal_best)
                    # print(np.where(np.array(num) != 0)[0])
                    #
                    # l = []
                    # for i in range(len(self.population)):
                    #     for j in range(len(self.population[i].personal_best)):
                    #         l2 = []
                    #         for k in range(3):
                    #             l2.append(
                    #                 list(abs(np.array(self.population[i].personal_best[j].fitness) - teste2[i][k])))
                    #         l2.sort()
                    #         l.extend(l2[0])
                    #         l3.append((i, j))
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    #
                    # l = []
                    # for i in range(len(self.population)):
                    #     l2 = abs(np.array(self.population[i].position) - teste3[i])
                    #     l.extend(l2)
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    #
                    # l = []
                    # for i in range(len(self.population)):
                    #     l2 = abs(np.array(self.population[i].fitness) - teste4[i])
                    #     l.extend(l2)
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    # if self.generation_count==7:
                    #     pass

                # teste nova memoria
                # print('teste nova memoria antes')
                # teste = np.zeros(
                #     self.params.position_dim * (2 * self.params.population_size
                #                                 + self.params.memory_size),
                #     dtype=np.float64)
                # cuda.memcpy_dtoh(teste, self.position_g)
                # teste.shape = 261, 10
                #
                # l = []
                # for i in range(len(self.memory)):
                #     l2 = np.array(self.memory[i].position) - teste[256 + i]
                #     l.extend(l2)
                # l3 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l3)
                #
                # teste2 = np.zeros(
                #     self.params.objectives_dim * (2 * self.params.population_size
                #                                   + self.params.memory_size),
                #     dtype=np.float64)
                # cuda.memcpy_dtoh(teste2, self.fitness_g)
                # teste2.shape = 261, 2
                #
                # l = []
                # for i in range(len(self.memory)):
                #     l2 = np.array(self.memory[i].fitness) - teste2[256 + i]
                #     l.extend(l2)
                # l3 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l3)

                start = dt.now()

                # se alguma particula for substituida pelo seu Xst
                # if self.update_from_differential_mutation:  # 0:00:03.856179 de 0:00:49.770765 7,73%
                #     # self.fronts = self.fast_nondominated_sort()
                #     # if cpu:
                #     self.fronts = self.fast_nondominated_sort()
                #
                #     # teste memory update1
                #     # front0_index = []
                #     # front0_part = []
                #     # for i in self.fronts[0]:
                #     #     for j in range(len(self.population)):
                #     #         if self.population[j] == i:
                #     #             front0_index.append(j)
                #     #             front0_part.append(self.population[j])
                #     #             break
                #     # print('front0', front0_index)
                #     # memory_prev = []
                #     # for i in range(len(self.memory)):
                #     #     front0_index.append(i+256)
                #     #     memory_prev.append(self.memory[i])
                #     # print('front0+mem', front0_index)
                #
                #     # else:
                #     # fronts_2 = self.fast_nondominated_sort_gpu()
                #     # self.fronts = self.fast_nondominated_sort_gpu()
                #
                #     # if self.generation_count==6:
                #     #     pass
                #
                #     self.memory_update()
                #
                #     # teste memory update2
                #     # f2 = []
                #     # for i in self.memory:
                #     #     for j in range(len(front0_part)):
                #     #         if front0_part[j] == i:
                #     #             f2.append(front0_index[j])
                #     #             break
                #     # print('front0', f2)
                #     # for i in self.memory:
                #     #     for j in range(len(memory_prev)):
                #     #         if memory_prev[j] == i:
                #     #             f2.append(j+256)
                #     #             break
                #     # print('front0', f2)
                #
                #     self.update_from_differential_mutation = False

                cpu[1] += (dt.now() - start).total_seconds()

                # if self.generation_count == 7:
                #     pass

                # teste_p = np.zeros(384 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 384, 10
                # teste_f = np.zeros(384 * 2, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 384, 2

                # # teste
                # teste_cur = np.zeros(1, np.int32)
                # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                # teste_f = np.zeros(256 * 3 * 3, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 256 * 3, 3
                # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                # plt.show()
                # pass

                # # teste
                # teste_cur = np.zeros(1, np.int32)
                # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                # teste_f = np.zeros(128*3*2, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 128*3, 2
                # teste_p = np.zeros(128 * 3 * 10, np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 128 * 3, 10
                # # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                # # plt.show()
                # pass

                if self.gpu:

                    start = dt.now()

                    # atualizar fronts
                    div = int(self.params.population_size/16)
                    fast_nondominated_sort = self.mod.get_function("fast_nondominated_sort")
                    fast_nondominated_sort(self.fitness_g, self.params.objectives_dim_g,
                                           self.domination_counter_g, self.params.population_size_g,
                                           self.params.otimizations_type_g, self.params.objectives_dim_g,
                                           block=(16, 32, 1),
                                           grid=(div, int(div/2), 1))
                    cuda.Context.synchronize()

                    #teste
                    # teste_f = np.zeros(384 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # teste_dc = np.zeros(385 * 385, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_dc, self.domination_counter_g)
                    # teste_dc.shape = 385, 385
                    # teste_dict[self.generation_count]['fit11'] = teste_f
                    # teste_dict[self.generation_count]['dc11'] = teste_dc

                    fast_nondominated_sort2 = self.mod.get_function("fast_nondominated_sort2")
                    fast_nondominated_sort2(self.domination_counter_g, self.params.population_size_g,
                                            self.params.population_size_g,
                                            block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    #teste
                    # teste_dc = np.zeros(385 * 385, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_dc, self.domination_counter_g)
                    # teste_dc.shape = 385, 385
                    # teste_dict[self.generation_count]['dc12'] = teste_dc

                    fast_nondominated_sort3 = self.mod.get_function("fast_nondominated_sort3")
                    fast_nondominated_sort3(self.domination_counter_g, self.params.population_size_g,
                                            self.params.population_size_g, self.fronts_g, self.tams_fronts_g,
                                            self.rank_g,
                                            block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # teste
                    # teste_front = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_front, self.fronts_g)
                    # teste_dict[self.generation_count]['f11'] = teste_front
                    # tam_front = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(tam_front, self.tams_fronts_g)
                    # teste_dict[self.generation_count]['tf11'] = tam_front
                    # pass

                    # teste fronts
                    # print("teste fronts")
                    # f = []
                    # for k in self.fronts:
                    #     l2 = []
                    #     for i in k:
                    #         for j in range(len(self.population)):
                    #             if self.population[j] == i:
                    #                 l2.append(j)
                    #                 break
                    #     l2.sort()
                    #     f.extend(l2)
                    # teste_front = np.zeros(128, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_front, self.fronts_g)
                    # diff = teste_front-f
                    # print(diff)
                    # if self.generation_count==7:
                    #     pass

                    # teste
                    # teste5 = np.zeros(128, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste5, self.fronts_g)
                    # teste6 = np.zeros(128, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste6, self.front0_mem_g)
                    # teste7 = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste7, self.tam_front0_mem_g)

                    #teste nan
                    # f_teste = open('testeNAN.pkl','wb')
                    # dict2 = {}

                    # atualiza memoria pela GPU
                    inicialize_front0_mem = self.mod.get_function("inicialize_front0_mem")
                    inicialize_front0_mem(self.fronts_g, self.front0_mem_g, self.tams_fronts_g,
                                          self.tam_front0_mem_g, self.position_g, self.params.memory_size_g,
                                          self.params.population_size_g,
                                          self.params.position_dim_g, self.params.current_memory_size_g,
                                          block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    #teste
                    # teste_cur = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                    # # teste_dict[self.generation_count]['cur21'] = teste_cur
                    # teste2 = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste2, self.fronts_g)
                    # # teste_dict[self.generation_count]['f21'] = teste2
                    # teste_tam = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                    # # teste_dict[self.generation_count]['tf21'] = teste_tam
                    # teste3 = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste3, self.front0_mem_g)
                    # # teste_dict[self.generation_count]['fronts0_mem_21'] = teste3
                    # teste4 = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste4, self.tam_front0_mem_g)
                    # # teste_dict[self.generation_count]['tam_fronts0_mem_21'] = teste4
                    # teste_p = np.zeros(384 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = 384, 10
                    # # teste_dict[self.generation_count]['pos21'] = teste_p
                    # teste_f = np.zeros(384 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # # teste_dict[self.generation_count]['fit21'] = teste_f
                    # pass

                    #teste nan
                    # teste_cur = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                    # dict2['cur'] = teste_cur
                    # teste2 = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste2, self.fronts_g)
                    # dict2['fronts'] = teste2
                    # teste_tam = np.zeros(128, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                    # dict2['tam_fronts'] = teste_tam
                    # teste3 = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste3, self.front0_mem_g)
                    # dict2['fronts0_mem'] = teste3
                    # teste4 = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste4, self.tam_front0_mem_g)
                    # dict2['tam_fronts0_mem'] = teste4
                    # teste_p = np.zeros(384 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = 384, 10
                    # dict2['pos'] = teste_p
                    # teste_f = np.zeros(384 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # dict2['fit'] = teste_f
                    # pickle.dump(dict2, f_teste)
                    # f_teste.close()

                    # teste
                    # teste2 = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste2, self.fronts_g)
                    # print(teste2[:15])
                    # teste3 = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste3, self.front0_mem_g)
                    # print(teste3[:15])
                    # teste4 = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste4, self.tam_front0_mem_g)
                    # print(teste4)
                    # teste_p = np.zeros(384 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = 384, 10
                    # teste_f = np.zeros(384 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # pass

                    # tam_front0_mem = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(tam_front0_mem, self.tam_front0_mem_g)
                    # fast_nondominated_sort4 = self.mod.get_function("fast_nondominated_sort4")
                    # fast_nondominated_sort4(self.fitness_g, self.params.objectives_dim_g,
                    #                         self.domination_counter_g, self.params.population_size_g,
                    #                         self.params.otimizations_type_g, self.params.objectives_dim_g,
                    #                         self.front0_mem_g, self.tam_front0_mem_g,
                    #                         block=(int(tam_front0_mem[0]), int(tam_front0_mem[0]), 1),
                    #                         grid=(1, 1, 1))
                    # cuda.Context.synchronize()

                    tam_front0_mem = np.zeros(1, dtype=np.int32)
                    cuda.memcpy_dtoh(tam_front0_mem, self.tam_front0_mem_g)
                    if tam_front0_mem > 32:
                        block_x = 32
                        grid_x = int(np.ceil(tam_front0_mem[0] / 32))
                    else:
                        block_x = int(tam_front0_mem[0])
                        grid_x = 1

                    # fast_nondominated_sort4 = self.mod.get_function("fast_nondominated_sort4")
                    # fast_nondominated_sort4(self.fitness_g, self.params.objectives_dim_g,
                    #                         self.domination_counter_g, self.params.population_size_g,
                    #                         self.params.otimizations_type_g, self.params.objectives_dim_g,
                    #                         self.front0_mem_g, self.tam_front0_mem_g,
                    #                         # block=(int(tam_front[0]), int(tam_front[0]), 1),
                    #                         # grid=(1, 1, 1))
                    #                         block=(block_x, block_x, 1),
                    #                         grid=(grid_x, grid_x, 1))
                    # cuda.Context.synchronize()

                    fast_nondominated_sort4_2 = self.mod.get_function("fast_nondominated_sort4_2")
                    fast_nondominated_sort4_2(self.fitness_g, self.params.objectives_dim_g,
                                            self.domination_counter_g, self.params.population_size_g,
                                            self.params.otimizations_type_g, self.params.objectives_dim_g,
                                            self.front0_mem_g, self.tam_front0_mem_g,
                                            # block=(int(tam_front[0]), int(tam_front[0]), 1),
                                            # grid=(1, 1, 1))
                                            block=(block_x, block_x, 1),
                                            grid=(grid_x, grid_x, 1))
                    cuda.Context.synchronize()

                    # teste
                    # teste_f = np.zeros(384 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # teste_dict[self.generation_count]['fit22'] = teste_f
                    # teste_d = np.zeros(385*385, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_d, self.domination_counter_g)
                    # teste_d.shape = 385, 385
                    # teste_dict[self.generation_count]['dc22'] = teste_d
                    # teste3 = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste3, self.front0_mem_g)
                    # teste_dict[self.generation_count]['fronts0_mem_22'] = teste3
                    # teste4 = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste4, self.tam_front0_mem_g)
                    # teste_dict[self.generation_count]['tam_fronts0_mem_22'] = teste4

                    # teste
                    # teste_d = np.zeros(25*25, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_d, self.domination_counter_g)
                    # teste_d.shape = 25, 25

                    fast_nondominated_sort5 = self.mod.get_function("fast_nondominated_sort5")
                    fast_nondominated_sort5(self.domination_counter_g,
                                            block=(int(tam_front0_mem[0]), 1, 1),
                                            grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    #teste
                    # teste_d = np.zeros(385 * 385, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_d, self.domination_counter_g)
                    # teste_d.shape = 385, 385
                    # teste_dict[self.generation_count]['dc23'] = teste_d

                    # teste
                    # teste2 = np.zeros(13 * 12, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste2, self.domination_counter_g)
                    # teste2.shape = 13, 12

                    # self.front0_g testeTempo250624 o front zero do conjunto front0 da populacao + memoria atual
                    fast_nondominated_sort6 = self.mod.get_function("fast_nondominated_sort6")
                    fast_nondominated_sort6(self.domination_counter_g, self.tam_front0_mem_g,
                                            self.front0_mem_g, self.tam_front0_g, self.front0_g,
                                            block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    #teste
                    # teste_d = np.zeros(385 * 385, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_d, self.domination_counter_g)
                    # teste_d.shape = 385, 385
                    # # teste_dict[self.generation_count]['dc24'] = teste_d
                    # teste5 = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste5, self.front0_mem_g)
                    # # teste_dict[self.generation_count]['fronts0_mem_24'] = teste3
                    # teste6 = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste6, self.tam_front0_mem_g)
                    # # teste_dict[self.generation_count]['tam_fronts0_mem_24'] = teste4
                    # teste3 = np.zeros(384, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste3, self.front0_g)
                    # # teste_dict[self.generation_count]['fronts0_24'] = teste3
                    # teste4 = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste4, self.tam_front0_g)
                    # # teste_dict[self.generation_count]['tam_fronts0_24'] = teste4
                    # pass

                    # teste
                    # teste_d = np.zeros(25 * 26, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_d, self.domination_counter_g)
                    # teste_d.shape = 26, 25
                    # teste_d2 = np.zeros(128, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_d2, self.front0_g)

                    i_g = cuda.mem_alloc(np.array([1], np.int32).nbytes) #cosnertar depois, colocar como atributo
                    tam_front0 = np.zeros(1, dtype=np.int32)
                    cuda.memcpy_dtoh(tam_front0, self.tam_front0_g)
                    if tam_front0[0] <= self.params.memory_size:
                        cuda.memcpy_htod(self.params.current_memory_size_g, tam_front0)
                        memory_inicialization2_1 = self.mod.get_function("memory_inicialization2_1")
                        # try:
                        memory_inicialization2_1(self.position_g, self.fitness_g, self.front0_g,
                                                 self.params.position_dim_g, self.params.objectives_dim_g,
                                                 self.params.population_size_g, self.aux_g, self.aux2_g,
                                                 block=(int(tam_front0[0]), 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                            #teste
                            # teste1 = 0
                            # teste_p = np.zeros(384 * 10, dtype=np.float64)
                            # cuda.memcpy_dtoh(teste_p, self.position_g)
                            # teste_p.shape = 384, 10
                            # teste_f = np.zeros(384 * 2, dtype=np.float64)
                            # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                            # teste_f.shape = 384, 2
                            # teste = np.zeros(384*10, dtype=np.float64)
                            # cuda.memcpy_dtoh(teste, self.aux_g)
                            # teste.shape = 384, 10
                            # pass

                        # except Exception as e:
                        #     teste_front = np.zeros(128, dtype=np.int32)
                        #     cuda.memcpy_dtoh(teste_front, self.fronts_g)
                        #     print('fronts')
                        #     print(teste_front)
                        #
                        #     teste_tam = np.zeros(128, dtype=np.int32)
                        #     cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                        #     print('tam fronts')
                        #     print(teste_tam)
                        #
                        #     teste = np.zeros(128*2, dtype=np.int32)
                        #     cuda.memcpy_dtoh(teste, self.front0_mem_g)
                        #     print('front 0 + mem')
                        #     print(teste)
                        #
                        #     teste = np.zeros(1, dtype=np.int32)
                        #     cuda.memcpy_dtoh(teste, self.tam_front0_mem_g)
                        #     print('tam front 0 mem')
                        #     print(teste)
                        #
                        #     teste = np.zeros(128 * 3, dtype=np.int32)
                        #     cuda.memcpy_dtoh(teste, self.front0_g)
                        #     print('front 0')
                        #     print(teste)
                        #
                        #     print('erro 2380')
                        #     print(tam_front0[0])
                        #     exit(0)

                        memory_inicialization2_2 = self.mod.get_function("memory_inicialization2_2")
                        memory_inicialization2_2(self.position_g, self.fitness_g, self.front0_g,
                                                 self.params.position_dim_g, self.params.objectives_dim_g,
                                                 self.params.population_size_g, self.aux_g, self.aux2_g,
                                                 block=(int(tam_front0[0]), 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()
                        # pass
                    else:
                        #teste
                        # teste1 = 1

                        # zerar vetor crowding distance
                        crowding_distance_inicialization = self.mod.get_function("crowding_distance_inicialization")
                        crowding_distance_inicialization(self.crowding_distance_g,
                                                         block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        # teste
                        # teste4 = np.zeros(261, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste4, self.crowding_distance_g)

                        for i in range(self.params.objectives_dim):
                            # ordena os fronts em ordem crescente de cada coordenada fitness
                            cuda.memcpy_htod(i_g, np.array([i], dtype=np.int32))
                            front_sort = self.mod.get_function("front_sort")
                            front_sort(self.front0_g, self.tam_front0_g, self.fitness_g,
                                       self.params.objectives_dim_g, i_g, self.crowding_distance_g,
                                       block=(1, 1, 1), grid=(1, 1, 1))
                            cuda.Context.synchronize()

                            # teste
                            # teste2 = np.zeros(128, dtype=np.int32)
                            # cuda.memcpy_dtoh(teste2, self.front0_g)
                            # teste3 = np.zeros(261*2, dtype=np.float64)
                            # cuda.memcpy_dtoh(teste3, self.fitness_g)
                            # teste3.shape = 261, 2
                            # print(teste2[:15])
                            # print(teste3[:7])
                            # print(teste3[-5:])

                            crowding_distance = self.mod.get_function("crowding_distance")
                            crowding_distance(self.front0_g, self.tam_front0_g, self.fitness_g,
                                              self.params.objectives_dim_g, self.tams_fronts_g, i_g,
                                              self.crowding_distance_g,
                                              block=(int(tam_front0[0]) - 2, 1, 1),
                                              grid=(1, 1, 1))

                            # teste
                            # teste4 = np.zeros(261, dtype=np.float64)
                            # cuda.memcpy_dtoh(teste4, self.crowding_distance_g)
                            # pass

                        front_sort_crowding_distance = self.mod.get_function("front_sort_crowding_distance")
                        front_sort_crowding_distance(self.front0_g, self.tam_front0_g,
                                                     self.crowding_distance_g,
                                                     block=(1, 1, 1), grid=(1, 1, 1))

                        # teste
                        # teste_pos = np.zeros(261*10, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_pos, self.position_g)
                        # teste_pos.shape = 261, 10
                        # teste_fit = np.zeros(261 * 2, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_fit, self.fitness_g)
                        # teste_fit.shape = 261, 2
                        # teste3 = np.zeros(261, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste3, self.crowding_distance_g)
                        # print(teste3[:10])
                        # teste4 = np.zeros(self.params.population_size, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste4, self.front0_g)
                        # print(teste4[:10])

                        # if self.generation_count == 0:
                        #     initial_memory_velocity1 = self.mod.get_function("initial_memory_velocity1")
                        #     initial_memory_velocity1(self.initial_memory_g, self.front0_g,
                        #                              block=(self.params.population_size, 1, 1),
                        #                              grid=(1, 1, 1))
                        #     cuda.Context.synchronize()

                        memory_inicialization2_1 = self.mod.get_function("memory_inicialization2_1")
                        memory_inicialization2_1(self.position_g, self.fitness_g, self.front0_g,
                                                 self.params.position_dim_g, self.params.objectives_dim_g,
                                                 self.params.population_size_g, self.aux_g, self.aux2_g,
                                                 block=(self.params.memory_size, 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        memory_inicialization2_2 = self.mod.get_function("memory_inicialization2_2")
                        memory_inicialization2_2(self.position_g, self.fitness_g, self.front0_g,
                                                 self.params.position_dim_g, self.params.objectives_dim_g,
                                                 self.params.population_size_g, self.aux_g, self.aux2_g,
                                                 block=(self.params.memory_size, 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                    gpu[1] += (dt.now() - start).total_seconds()

                    # teste
                    # teste_cur = np.zeros(1, np.int32)
                    # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                    # teste_f = np.zeros(384 * 2, np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # teste_p = np.zeros(384 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = 384, 10
                    # # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                    # # plt.show()
                    # pass

                    # teste_nan
                    # teste_f = np.zeros(384*2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # teste_p = np.zeros(384 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = 384, 10
                    # a = np.isnan(teste_f)
                    # a = sum(sum(a))
                    # if a > 0:
                    #     print(self.generation_count)
                    #     a = np.where(np.isnan(teste_f) == True)
                    #     exit(1)

                    # trecho usado apenas para igualar as duas simulações, apagar depois
                    # teste = np.zeros(
                    #     self.params.position_dim * (2 * self.params.population_size
                    #                                 + self.params.memory_size),
                    #     dtype=np.float64)
                    # cuda.memcpy_dtoh(teste, self.position_g)
                    # teste.shape = 2 * self.params.population_size + self.params.memory_size, self.params.position_dim
                    # teste2 = np.zeros(
                    #     self.params.objectives_dim * (2 * self.params.population_size + self.params.memory_size),
                    #     dtype=np.float64)
                    # cuda.memcpy_dtoh(teste2, self.fitness_g)
                    # teste2.shape = 2 * self.params.population_size + self.params.memory_size, self.params.objectives_dim
                    # if abs(self.memory[0].position[0] - teste[2 * self.params.population_size][0]) > 1e-3:
                    #     # print("troca2")
                    #     temp = teste[2 * self.params.population_size].copy()
                    #     teste[2 * self.params.population_size] = teste[2 * self.params.population_size + 1].copy()
                    #     teste[2 * self.params.population_size + 1] = temp.copy()
                    #     temp = teste2[2 * self.params.population_size].copy()
                    #     teste2[2 * self.params.population_size] = teste2[2 * self.params.population_size + 1].copy()
                    #     teste2[2 * self.params.population_size + 1] = temp.copy()
                    # cuda.memcpy_htod(self.position_g, teste.flatten())
                    # cuda.memcpy_htod(self.fitness_g, teste2.flatten())

                    # teste nova memoria
                    # print('teste nova memoria depois')
                    # teste = np.zeros(
                    #     self.params.position_dim * (2 * self.params.population_size
                    #                                 + self.params.memory_size),
                    #     dtype=np.float64)
                    # cuda.memcpy_dtoh(teste, self.position_g)
                    # teste.shape = 261, 10
                    #
                    # l = []
                    # for i in range(len(self.memory)):
                    #     l2 = np.array(self.memory[i].position) - teste[256 + i]
                    #     l.extend(l2)
                    # l3 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l3)
                    #
                    # teste2 = np.zeros(
                    #     self.params.objectives_dim * (2 * self.params.population_size
                    #                                   + self.params.memory_size),
                    #     dtype=np.float64)
                    # cuda.memcpy_dtoh(teste2, self.fitness_g)
                    # teste2.shape = 261, 2
                    #
                    # l = []
                    # for i in range(len(self.memory)):
                    #     l2 = np.array(self.memory[i].fitness) - teste2[256 + i]
                    #     l.extend(l2)
                    # l3 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l3)
                    # if self.generation_count==7:
                    #     pass

                start = dt.now()

                # copia pesos testeTempo250624 particulas
                # if self.copy_pop:  # 0:00:13.549243 de 0:01:01.481072 22,02%
                #     self.population_copy = copy.deepcopy(self.population)
                #     self.weights_copy = copy.deepcopy(self.weights)

                cpu[2] += (dt.now() - start).total_seconds()
                start = dt.now()

                if self.gpu:
                    # # teste - copiar mesmo teste depois da copia
                    # print('posicoes antes copia')

                    # copia de position
                    div = int(self.params.population_size/64)
                    copy2 = self.mod.get_function("copy")
                    copy2(self.position_g,
                          block=(int(self.params.population_size / div), self.params.position_dim, 1),
                          grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # teste
                    # print('posicoes apos copia')
                    # teste_pos = np.zeros(261 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_pos, self.position_g)
                    # teste_pos.shape = 261, 10
                    # l = []
                    # for i in range(len(self.population)):
                    #     for j in range(len(self.population[i].position)):
                    #         l.append(abs(self.population[i].position[j]-teste_pos[i][j]))
                    # for i in range(len(self.population_copy)):
                    #     for j in range(len(self.population_copy[i].position)):
                    #         l.append(abs(self.population_copy[i].position[j]-teste_pos[i+128][j]))
                    # l = np.array(l)
                    # l = np.where(l>1e-5)[0]
                    # print(l)
                    # if self.generation_count == 7:
                    #     pass

                    # teste - copiar mesmo teste depois da copia
                    # print('fitness antes copia')

                    # copia de fitness
                    copy2(self.fitness_g,
                          block=(int(self.params.population_size / div), self.params.objectives_dim, 1),
                          grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # teste
                    # print('fitness apos copia')
                    # teste_fit = np.zeros(261 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_fit, self.fitness_g)
                    # teste_fit.shape = 261, 2
                    # l_f = []
                    # for i in range(len(self.population)):
                    #     for j in range(len(self.population[i].fitness)):
                    #         l_f.append(abs(self.population[i].fitness[j] - teste_fit[i][j]))
                    # for i in range(len(self.population_copy)):
                    #     for j in range(len(self.population_copy[i].fitness)):
                    #         l_f.append(abs(self.population_copy[i].fitness[j] - teste_fit[i + 128][j]))
                    # l_f = np.array(l_f)
                    # l_f = np.where(l_f > 1e-5)[0]
                    # print(l_f)

                    # copia de rank
                    copy2(self.rank_g,
                          block=(int(self.params.population_size / div), 1, 1),
                          grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # teste
                    # print('velocidade antes copia')

                    # copia de velocity
                    copy2(self.velocity_g,
                          block=(int(self.params.population_size / div), self.params.position_dim, 1),
                          grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # teste
                    # print('velocidade depois copia')
                    # teste_v = np.zeros(261 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_v, self.velocity_g)
                    # teste_v.shape = 261, 10
                    # l2 = []
                    # for i in range(len(self.population)):
                    #     for j in range(len(self.population[i].velocity)):
                    #         l2.append(abs(self.population[i].velocity[j] - teste_v[i][j]))
                    # for i in range(len(self.population_copy)):
                    #     for j in range(len(self.population_copy[i].velocity)):
                    #         l2.append(abs(self.population_copy[i].velocity[j] - teste_v[i + 128][j]))
                    # l2 = np.array(l2)
                    # l2 = np.where(l2 > 1e-5)[0]
                    # print(l2)

                    # copia de personal_best
                    # div = 8
                    div = int(self.params.population_size/16)
                    copy2(self.personal_best_position_g,
                          block=(int(self.params.population_size / div),
                                 self.params.position_dim * self.params.personal_guide_array_size, 1),
                          grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # div = 2
                    div = int(self.params.population_size / 64)
                    copy2(self.personal_best_fitness_g,
                          block=(int(self.params.population_size / div),
                                 self.params.objectives_dim * self.params.personal_guide_array_size, 1),
                          grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # teste
                    # print('weights antes copia')

                    # copia de weights
                    # div = 1
                    div = int(self.params.population_size/128)
                    copy2 = self.mod.get_function("copy2")
                    copy2(self.weights_g, self.weights_copy_g,
                          block=(6, int(self.params.population_size / div), 1),
                          grid=(1, div, 1))
                    cuda.Context.synchronize()

                    # teste
                    # print('weights depois copia')
                    # teste_w = np.zeros(6 * 128, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_w, self.weights_g)
                    # teste_w.shape = 6,128
                    # l3 = []
                    # for i in range(len(self.weights)):
                    #     for j in range(len(self.weights[i])):
                    #         l3.append(abs(self.weights[i][j] - teste_w[i][j]))
                    # l3 = np.array(l3)
                    # l3 = np.where(l3 > 1e-5)[0]
                    # print(l3)
                    # teste_wc = np.zeros(6 * 128, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_wc, self.weights_copy_g)
                    # teste_wc.shape = 6, 128
                    # l2 = []
                    # for i in range(len(self.weights)):
                    #     for j in range(len(self.weights_copy[i])):
                    #         l2.append(abs(self.weights_copy[i][j] - teste_wc[i][j]))
                    # l2 = np.array(l2)
                    # l2 = np.where(l2 > 1e-5)[0]
                    # print(l2)

                gpu[2] += (dt.now() - start).total_seconds()

                # teste
                # teste_cur = np.zeros(1, np.int32)
                # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                # teste_f = np.zeros(256 * 3 * 3, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 256 * 3, 3
                # teste_p = np.zeros(384 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 384, 10
                # teste_v = np.zeros(384 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_v, self.velocity_g)
                # teste_v.shape = 384, 10
                # teste_w = np.zeros(6 * 128, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_w, self.weights_g)
                # teste_w.shape = 6, 128
                # teste_wc = np.zeros(6 * 128, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_wc, self.weights_copy_g)
                # teste_wc.shape = 6, 128
                # # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                # # plt.show()
                # pass

                start = dt.now()

                ## muta os pesos de ambas populacoes - 10%
                # np.random.seed(0)
                # self.mutate_weights()  # 0:00:06.260588 de  0:00:50.126215 4,5%

                cpu[3] += (dt.now() - start).total_seconds()

                # para manter a igualdade das implementacoes, simplesmente copiei os novos vetores
                # mais tarde elembrar de  impleementar essa mutação via gpu
                if self.gpu:
                    start = dt.now()
                    # weights = np.zeros(6 * 128, dtype=np.float64)
                    weights = np.zeros(6 * self.params.population_size, dtype=np.float64)
                    cuda.memcpy_htod(self.weights_g, weights)

                    div = int(self.params.population_size/128)
                    cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))

                    mutate_weights = self.mod.get_function("mutate_weights")
                    mutate_weights(self.weights_g, self.seed_g, self.params.population_size_g,
                                   self.params.mutation_rate_g,
                          block=(6, int(self.params.population_size/div), 1),
                          grid=(1, div, 1))
                    cuda.Context.synchronize()

                    mutate_weights2 = self.mod.get_function("mutate_weights2")
                    mutate_weights2(self.weights_g, self.params.population_size_g,
                                   block=(4, int(self.params.population_size/div), 1),
                                   grid=(1, div, 1))
                    cuda.Context.synchronize()

                    mutate_weights3 = self.mod.get_function("mutate_weights3")
                    mutate_weights3(self.weights_g, self.params.population_size_g,
                                    block=(int(self.params.population_size), 1, 1),
                                    grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    #weights copy
                    cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))
                    mutate_weights(self.weights_copy_g, self.seed_g, self.params.population_size_g,
                                   self.params.mutation_rate_g,
                                   block=(6, int(self.params.population_size/div), 1),
                                   grid=(1, div, 1))
                    cuda.Context.synchronize()

                    mutate_weights2(self.weights_copy_g, self.params.population_size_g,
                                    block=(4, int(self.params.population_size/div), 1),
                                    grid=(1, div, 1))
                    cuda.Context.synchronize()

                    mutate_weights3(self.weights_copy_g, self.params.population_size_g,
                                    block=(int(self.params.population_size), 1, 1),
                                    grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # mutate_weights(self.weights_copy_g,
                    #                block=(6, int(self.params.population_size), 1),
                    #                grid=(1, 1, 1))
                    # cuda.Context.synchronize()

                    #teste
                    # weights = np.zeros(6 * 128, dtype=np.float64)
                    # cuda.memcpy_dtoh(weights, self.weights_g)
                    # weights.shape = 6, 128
                    # for i in range(6):
                    #     plt.figure()
                    #     plt.hist(weights[i])
                    #     plt.show()
                    # weights = np.zeros(6 * 128, dtype=np.float64)
                    # cuda.memcpy_dtoh(weights, self.weights_copy_g)
                    # weights.shape = 6, 128
                    # for i in range(6):
                    #     plt.figure()
                    #     plt.hist(weights[i])
                    #     plt.show()
                    #
                    # pass

                    # weights = np.zeros(6 * 128, dtype=np.float64)
                    # cuda.memcpy_dtoh(weights, self.weights_copy_g)
                    # # weights.shape = 6, 128
                    # print(sum(np.bitwise_and(weights > -2, weights < 2)) / (128 * 6))
                    # print(sum(np.bitwise_and(weights > -3, weights < 3)) / (128 * 6))
                    # plt.subplot(2, 2, 3)
                    # plt.hist(weights)
                    # plt.show()

                    # cuda.memcpy_htod(self.weights_g, self.weights.astype(np.float64).flatten())
                    # cuda.memcpy_htod(self.weights_copy_g, self.weights_copy.astype(np.float64).flatten())

                    gpu[3] += (dt.now() - start).total_seconds()
                    # cuda.memcpy_htod(self.position_g, teste.flatten())
                    # cuda.memcpy_htod(self.fitness_g, teste2.flatten())

                # teste
                # print('teste apos mutacao pesos')
                # teste = np.zeros(6 * 128, dtype=np.float64)
                # cuda.memcpy_dtoh(teste, self.weights_g)
                # teste.shape = 6, 128
                # l = []
                # for i in range(len(self.weights)):
                #     for j in range(len(self.weights[i])):
                #         l.append(abs(self.weights[i][j] - teste[i][j]))
                # l = np.array(l)
                # l = np.where(l > 1e-5)[0]
                # print('teste weigths', l)
                #
                # teste = np.zeros(6 * 128, dtype=np.float64)
                # cuda.memcpy_dtoh(teste, self.weights_copy_g)
                # teste.shape = 6, 128
                # l = []
                # for i in range(len(self.weights)):
                #     for j in range(len(self.weights[i])):
                #         l.append(abs(self.weights_copy[i][j] - teste[i][j]))
                # l = np.array(l)
                # l = np.where(l > 1e-5)[0]
                # print('teste weights copia', l)
                # if self.generation_count == 7:
                #     pass

                start = dt.now()
                ## Atualizar melhores globais. 17.45 de 47.22 testeTempo250624 0:00:17.199883 de 0:00:45.820871 37,53%
                # if self.copy_pop:
                #     self.global_best_attribution(True)
                #     # self.global_best_attribution_gpu(True)
                # else:
                #     self.global_best_attribution()

                cpu[4] += (dt.now() - start).total_seconds()
                start = dt.now()

                if self.gpu:
                    div = int(np.ceil((self.params.population_size * 2 + self.params.memory_size)/512))
                    sigma_eval = self.mod.get_function("sigma_eval")
                    sigma_eval(self.sigma_g, self.fitness_g, self.params.objectives_dim_g,
                               block=(int((self.params.population_size * 2 + self.params.memory_size)/div), 1, 1),
                               grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # teste sigma
                    # print('teste sigma')
                    # teste_s = np.zeros(261 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_s, self.sigma_g)
                    # teste_s.shape = 261, 2
                    # l2 = []
                    # for i in range(len(self.population)):
                    #     l2.append(abs(self.population[i].sigma_value - teste_s[i][0]))
                    # for i in range(len(self.population_copy)):
                    #     l2.append(abs(self.population_copy[i].sigma_value - teste_s[i + 128][0]))
                    # for i in range(len(self.memory)):
                    #     l2.append(abs(self.memory[i].sigma_value - teste_s[i + 256][0]))
                    # l2 = np.array(l2)
                    # l2 = np.where(l2 > 1e-5)[0]
                    # print(l2)
                    # if self.generation_count == 7:
                    #     pass

                    sigma_nearest = self.mod.get_function("sigma_nearest")
                    sigma_nearest(self.sigma_g, self.fronts_g, self.tams_fronts_g, self.rank_g,
                                  self.params.population_size_g, self.params.memory_size_g,
                                  self.params.objectives_dim_g, self.global_best_g, self.fitness_g,
                                  block=(int(2 * self.params.population_size), 1, 1),
                                  grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # teste fronts
                    # f = []
                    # for k in self.fronts:
                    #     l2 = []
                    #     for i in k:
                    #         for j in range(len(self.population)):
                    #             if self.population[j] == i:
                    #                 l2.append(j)
                    #                 break
                    #     l2.sort()
                    #     f.extend(l2)
                    # teste_front = np.zeros(128, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_front, self.fronts_g)
                    # teste_tf = np.zeros(128, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_tf, self.tams_fronts_g)

                    # retirar depois. Em alguns casos particulas com a mesma distancia sigma eram escolhidos
                    # teste global_best
                    # de forma diferente pela cpu testeTempo250624 gpu. Nesses casos foia dotado o da gpu para validacao
                    # print('teste global best')
                    # # teste_f = np.zeros(261*2, dtype=np.float64)
                    # # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # # teste_f.shape = 261, 2
                    # teste_g = np.zeros(261, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_g, self.global_best_g)
                    # l2 = []
                    # for i in range(len(self.population)):
                    #     if self.population[i].rank == 0:
                    #         l2.append(np.where(np.array(self.memory) == self.population[i].global_best)[0][0])
                    #         l2[-1] += 256
                    #     else:
                    #         l2.append(np.where(np.array(self.population) == self.population[i].global_best)[0][0])
                    # for i in range(len(self.population_copy)):
                    #     if self.population_copy[i].rank == 0:
                    #         l2.append(np.where(np.array(self.memory) == self.population_copy[i].global_best)[0][0])
                    #         l2[-1] += 256
                    #     else:
                    #         l2.append(np.where(np.array(self.population) == self.population_copy[i].global_best)[0][0])
                    # # print(l2 - teste_g[:256])
                    # # print(np.where((l2 - teste_g[:256]) != 0))
                    # l3 = np.where((l2 - teste_g[:256]) != 0)[0]
                    # if len(l3) > 0 :
                    #     print(l3)
                    #     teste_s = np.zeros(261 * 2, dtype=np.float64)
                    #     cuda.memcpy_dtoh(teste_s, self.sigma_g)
                    #     teste_s.shape = 261, 2
                    #     for i in l3:
                    #         a = teste_g[i]
                    #         a = teste_s[a]
                    #         # b = np.where((np.array(self.population) == self.population[82].global_best)==True)[0][0]
                    #         if i<128:
                    #             b = self.population[i].global_best.sigma_value
                    #             if abs(b - a[0]) < 1e-5:
                    #                 try:
                    #                     b = \
                    #                     np.where((np.array(self.population) == self.population[i].global_best) == True)[0][
                    #                         0]
                    #                 except IndexError:
                    #                     print(b, a, b-a[0], i)
                    #                 else:
                    #                     teste_g[i] = b
                    #         else:
                    #             b = self.population_copy[i-128].global_best.sigma_value
                    #             if abs(b-a[0])<1e-5:
                    #                 try:
                    #                     b = np.where((np.array(self.population) == self.population_copy[i-128].global_best) == True)[0][
                    #                         0]
                    #                 except IndexError:
                    #                     print(b, a, b - a[0], i)
                    #                 else:
                    #                     teste_g[i] = b
                    #     cuda.memcpy_htod(self.global_best_g, teste_g)
                    #
                    #     print('teste global best apos troca')
                    #     teste_g = np.zeros(261, dtype=np.int32)
                    #     cuda.memcpy_dtoh(teste_g, self.global_best_g)
                    #     l2 = []
                    #     for i in range(len(self.population)):
                    #         if self.population[i].rank == 0:
                    #             l2.append(np.where(np.array(self.memory) == self.population[i].global_best)[0][0])
                    #             l2[-1] += 256
                    #         else:
                    #             l2.append(np.where(np.array(self.population) == self.population[i].global_best)[0][0])
                    #     for i in range(len(self.population_copy)):
                    #         if self.population_copy[i].rank == 0:
                    #             l2.append(np.where(np.array(self.memory) == self.population_copy[i].global_best)[0][0])
                    #             l2[-1] += 256
                    #         else:
                    #             l2.append(
                    #                 np.where(np.array(self.population) == self.population_copy[i].global_best)[0][0])
                    #     # print(l2 - teste_g[:256])
                    #     print(np.where((l2 - teste_g[:256]) != 0)[0])
                    # else:
                    #     print(l3)
                    # pass

                gpu[4] += (dt.now() - start).total_seconds()

                # if self.generation_count == 7:
                #     pass
                # teste pb
                # print('\nteste antes pb move')
                # teste = np.zeros(261 * 30, dtype=np.float64)
                # cuda.memcpy_dtoh(teste, self.personal_best_position_g)
                # teste.shape = 261, 3, 10
                # teste2 = np.zeros(261 * 6, dtype=np.float64)
                # cuda.memcpy_dtoh(teste2, self.personal_best_fitness_g)
                # teste2.shape = 261, 3, 2
                # teste3 = np.zeros(261 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste3, self.position_g)
                # teste3.shape = 261, 10
                # teste4 = np.zeros(261 * 2, dtype=np.float64)
                # cuda.memcpy_dtoh(teste4, self.fitness_g)
                # teste4.shape = 261, 2
                # if self.generation_count == 7:
                #     pass
                #
                # l = []
                # l3 = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(abs(np.array(self.population[i].personal_best[j].position) - teste[i][k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # num = []
                # for i in range(len(self.population)):
                #     num.append(0)
                #     for j in range(3):
                #         if teste[i][j][0] != 1e10:
                #             num[-1] += 1
                #     num[-1] -= len(self.population[i].personal_best)
                # print(np.where(np.array(num) != 0)[0])
                #
                # if self.generation_count == 7:
                #     pass
                #
                # l = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(abs(np.array(self.population[i].personal_best[j].fitness) - teste2[i][k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # for i in range(len(self.population)):
                #     l2 = abs(np.array(self.population[i].position) - teste3[i])
                #     l.extend(l2)
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # for i in range(len(self.population)):
                #     l2 = abs(np.array(self.population[i].fitness) - teste4[i])
                #     l.extend(l2)
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # l3 = []
                # for i in range(len(self.population_copy)):
                #     for j in range(len(self.population_copy[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(
                #                     abs(np.array(self.population_copy[i].personal_best[j].position) - teste[i + 128][
                #                         k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                # if self.generation_count == 7:
                #     pass
                #
                # l = []
                # for i in range(len(self.population_copy)):
                #     for j in range(len(self.population_copy[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(
                #                     abs(np.array(self.population_copy[i].personal_best[j].fitness) - teste2[i + 128][
                #                         k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # for i in range(len(self.population_copy)):
                #     l2 = abs(np.array(self.population_copy[i].position) - teste3[i + 128])
                #     l.extend(l2)
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # for i in range(len(self.population_copy)):
                #     l2 = abs(np.array(self.population_copy[i].fitness) - teste4[i + 128])
                #     l.extend(l2)
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                # if self.generation_count == 7:
                #     pass

                start = dt.now()

                ## Aplica movimento em todas particulas 0:00:00.828291 de 0:00:58.617058 1,39%
                # for i, p in enumerate(self.population):
                #     if i == 7:
                #         self.move_particle(p, i, False)
                #     else:
                #         self.move_particle(p, i, False)
                #     if i == 29:
                #         #desfazer a alteracao depois na dominacao, para disconsiderar diferenças
                #         # abaixo de 1e-6
                #         self.update_personal_best(p)
                #     else:
                #         self.update_personal_best(p)
                #
                # if self.copy_pop:
                #     for i, p in enumerate(self.population_copy):
                #         if i == 7:
                #             self.move_particle(p, i, True)
                #         else:
                #             self.move_particle(p, i, True)
                #         self.update_personal_best(p)

                cpu[5] += (dt.now() - start).total_seconds()
                start = dt.now()

                if self.gpu:
                    # div = 2
                    div = int(self.params.population_size/64)
                    # cuda.memcpy_htod(self.whatPersonal_g, self.whatPersonal)
                    # cuda.memcpy_htod(self.communication_g, self.communication)
                    # cuda.memcpy_htod(self.cooperation_rand_g, self.cooperation_rand)
                    # teste2 = np.zeros(261, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste2, self.cooperation_rand_g)
                    # print(teste2[:5])
                    # print(self.cooperation_rand[:5])
                    # pass

                    cuda.memcpy_htod(self.seed_g, np.random.randint(0, int(1e9), dtype=np.int32))
                    # move_particle = self.mod.get_function("move_particle")
                    # move_particle(self.weights_g, self.weights_copy_g, self.personal_best_position_g,
                    #               self.position_g, self.velocity_g, self.whatPersonal_g,
                    #               self.params.personal_guide_array_size_g,
                    #               self.communication_g, self.cooperation_rand_g,
                    #               self.params.communication_probability_g,
                    #               self.global_best_g, self.params.velocity_max_value_g,
                    #               self.params.velocity_min_value_g, self.seed_g,
                    #               block=(int(self.params.population_size / div), self.params.position_dim, 1),
                    #               grid=(div, 1, 1))
                    # cuda.Context.synchronize()
                    move_particle = self.mod.get_function("move_particle")
                    move_particle(self.weights_g, self.weights_copy_g, self.personal_best_position_g,
                                  self.position_g, self.velocity_g,
                                  self.params.personal_guide_array_size_g,
                                  self.params.communication_probability_g,
                                  self.global_best_g, self.params.velocity_max_value_g,
                                  self.params.velocity_min_value_g, self.seed_g,
                                  block=(int(self.params.population_size / div), self.params.position_dim, 1),
                                  grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    div *= 2
                    move_particle2 = self.mod.get_function("move_particle2")
                    move_particle2(self.position_g, self.velocity_g, self.params.position_min_value_g,
                                   self.params.position_max_value_g,
                                   block=(int(2 * self.params.population_size / div), self.params.position_dim, 1),
                                   grid=(div, 1, 1))
                    cuda.Context.synchronize()

                    # teste velocity
                    # print('teste depois velocity')

                    # teste position
                    # print('teste depois position')

                    # if self.generation_count==0:
                    #     initial_memory_velocity2 = self.mod.get_function("initial_memory_velocity2")
                    #     initial_memory_velocity2(self.velocity_g, self.initial_memory_g,
                    #                              self.params.population_size_g,
                    #                              self.params.position_dim_g,
                    #                    block=(self.params.memory_size, 1, 1),
                    #                    grid=(1, 1, 1))
                    #     cuda.Context.synchronize()
                    #
                    #     teste3 = np.zeros(261*10, dtype=np.float64)
                    #     cuda.memcpy_dtoh(teste3, self.velocity_g)
                    #     teste3.shape = 261, 10
                    #     teste4 = np.zeros(128, dtype=np.int32)
                    #     cuda.memcpy_dtoh(teste4, self.initial_memory_g)
                    #     pass

                    # if self.generation_count==0:
                    #     initial_memory_velocity = self.mod.get_function("initial_memory_velocity")
                    #     initial_memory_velocity(self.velocity_g, self.params.population_size_g,
                    #                    block=(self.params.memory_size, self.params.position_dim, 1),
                    #                    grid=(1, 1, 1))
                    #     cuda.Context.synchronize()

                    # teste fitness
                    # print('teste antes fitness')
                    # teste_f = np.zeros(261 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 261, 2
                    # l2 = []
                    # for i in range(len(self.population)):
                    #     for j in range(len(self.population[i].fitness)):
                    #         l2.append(abs(self.population[i].fitness[j] - teste_f[i][j]))
                    # for i in range(len(self.population_copy)):
                    #     for j in range(len(self.population_copy[i].fitness)):
                    #         l2.append(abs(self.population_copy[i].fitness[j] - teste_f[i + 128][j]))
                    # l2 = np.array(l2)
                    # l2 = np.where(l2 > 1e-5)[0]
                    # print(l2)
                    # pass

                    # if func == 'ZDT1':
                    # zdt1 = self.mod.get_function("zdt1")
                    function = self.mod.get_function("function")
                    function(self.params.func_n_g, self.position_g, self.params.position_dim_g,
                             self.fitness_g, self.alpha_g,
                         block=(2 * self.params.population_size, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()
                    self.fitness_eval_count += 2*self.params.population_size

                    # teste fitness
                    # print('teste depois fitness')

                    # update_personal_best3 = self.mod.get_function("update_personal_best3")
                    update_personal_best3 = self.mod.get_function("update_personal_best3_validation")
                    update_personal_best3(self.personal_best_position_g, self.personal_best_velocity_g,
                                          self.personal_best_fitness_g,
                                          self.params.objectives_dim_g, self.params.position_dim_g, self.position_g,
                                          self.fitness_g,
                                          self.params.personal_guide_array_size_g,
                                          self.params.otimizations_type_g,
                                          block=(2 * self.params.population_size, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # teste pb
                    # print('\nteste depois pb move')
                    # teste = np.zeros(261 * 30, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste, self.personal_best_position_g)
                    # teste.shape = 261, 3, 10
                    # teste2 = np.zeros(261 * 6, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste2, self.personal_best_fitness_g)
                    # teste2.shape = 261, 3, 2
                    # teste3 = np.zeros(261 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste3, self.position_g)
                    # teste3.shape = 261, 10
                    # teste_f = np.zeros(256 * 3 * 3, np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 256 * 3, 3
                    # pass
                    # if self.generation_count == 7:
                    #     pass
                    #
                    # l = []
                    # l3 = []
                    # for i in range(len(self.population)):
                    #     for j in range(len(self.population[i].personal_best)):
                    #         l2 = []
                    #         for k in range(3):
                    #             l2.append(
                    #                 list(abs(np.array(self.population[i].personal_best[j].position) - teste[i][k])))
                    #         l2.sort()
                    #         l.extend(l2[0])
                    #         l3.append((i, j))
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    #
                    # num = []
                    # for i in range(len(self.population)):
                    #     num.append(0)
                    #     for j in range(3):
                    #         if teste[i][j][0] != 1e10:
                    #             num[-1] += 1
                    #     num[-1] -= len(self.population[i].personal_best)
                    # print(np.where(np.array(num) != 0)[0])
                    #
                    # if self.generation_count == 7:
                    #     pass
                    #
                    # l = []
                    # for i in range(len(self.population)):
                    #     for j in range(len(self.population[i].personal_best)):
                    #         l2 = []
                    #         for k in range(3):
                    #             l2.append(
                    #                 list(abs(np.array(self.population[i].personal_best[j].fitness) - teste2[i][k])))
                    #         l2.sort()
                    #         l.extend(l2[0])
                    #         l3.append((i, j))
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    #
                    # l = []
                    # for i in range(len(self.population)):
                    #     l2 = abs(np.array(self.population[i].position) - teste3[i])
                    #     l.extend(l2)
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    #
                    # l = []
                    # for i in range(len(self.population)):
                    #     l2 = abs(np.array(self.population[i].fitness) - teste4[i])
                    #     l.extend(l2)
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    #
                    # l = []
                    # l3 = []
                    # for i in range(len(self.population_copy)):
                    #     for j in range(len(self.population_copy[i].personal_best)):
                    #         l2 = []
                    #         for k in range(3):
                    #             l2.append(
                    #                 list(
                    #                     abs(np.array(self.population_copy[i].personal_best[j].position) - teste[i+128][k])))
                    #         l2.sort()
                    #         l.extend(l2[0])
                    #         l3.append((i, j))
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    # if self.generation_count == 6:
                    #     pass
                    #
                    # l = []
                    # for i in range(len(self.population_copy)):
                    #     for j in range(len(self.population_copy[i].personal_best)):
                    #         l2 = []
                    #         for k in range(3):
                    #             l2.append(
                    #                 list(
                    #                     abs(np.array(self.population_copy[i].personal_best[j].fitness) - teste2[i+128][k])))
                    #         l2.sort()
                    #         l.extend(l2[0])
                    #         l3.append((i, j))
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    #
                    # l = []
                    # for i in range(len(self.population_copy)):
                    #     l2 = abs(np.array(self.population_copy[i].position) - teste3[i+128])
                    #     l.extend(l2)
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    #
                    # l = []
                    # for i in range(len(self.population_copy)):
                    #     l2 = abs(np.array(self.population_copy[i].fitness) - teste4[i+128])
                    #     l.extend(l2)
                    # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                    # print(l2)
                    # if self.generation_count == 7:
                    #     pass

                gpu[5] += (dt.now() - start).total_seconds()
                start = dt.now()

                # # teste
                # teste_cur = np.zeros(1, np.int32)
                # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                # teste_f = np.zeros(384 * 2, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 384, 2
                # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                # plt.show()
                # pass

                ## Separar particulas em fronteiras.
                # if self.copy_pop:  # 0:00:05.484410 de 0:01:00.081699 9,12%
                #     self.fronts = self.fast_nondominated_sort(False, True)
                # else:
                #     self.fronts = self.fast_nondominated_sort(False)

                cpu[6] += (dt.now() - start).total_seconds()
                start = dt.now()

                if self.gpu:
                    # atualizar fronts
                    div1 = int(2*self.params.population_size/16)
                    div2 = int(self.params.population_size * 2 / 32)
                    fast_nondominated_sort = self.mod.get_function("fast_nondominated_sort")
                    fast_nondominated_sort(self.fitness_g, self.params.objectives_dim_g,
                                           self.domination_counter_g, self.params.population_size_g,
                                           self.params.otimizations_type_g, self.params.objectives_dim_g,
                                           block=(16, 32, 1),
                                           grid=(div1, div2, 1))
                    cuda.Context.synchronize()

                    # temp = cuda.mem_alloc(np.zeros(1, dtype=np.int32).nbytes)
                    # cuda.memcpy_htod(temp, np.array(2 * self.params.population_size, dtype=np.int32))
                    #
                    # fast_nondominated_sort = self.mod.get_function("fast_nondominated_sort")
                    # fast_nondominated_sort(self.fitness_g, self.params.objectives_dim_g,
                    #                        self.domination_counter_g, temp,
                    #                        self.params.otimizations_type_g, self.params.objectives_dim_g,
                    #                        block=(16, 32, 1),
                    #                        grid=(16, 8, 1))
                    # cuda.Context.synchronize()

                    #teste
                    # teste_f = np.zeros(256 * 3 * 3, np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 256 * 3, 3
                    # teste_dc = np.zeros(385 * 385, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_dc, self.domination_counter_g)
                    # teste_dc.shape = 385, 385
                    # teste_dict[self.generation_count]['fit31'] = teste_f
                    # teste_dict[self.generation_count]['dc31'] = teste_dc

                    # teste = np.zeros(256**2, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste, self.domination_counter_g)
                    # teste.shape = 256, 256

                    temp = cuda.mem_alloc(np.zeros(1, dtype=np.int32).nbytes)
                    cuda.memcpy_htod(temp, np.array(2 * self.params.population_size, dtype=np.int32))
                    fast_nondominated_sort2 = self.mod.get_function("fast_nondominated_sort2")
                    fast_nondominated_sort2(self.domination_counter_g, temp, temp,
                                            block=(2 * self.params.population_size, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    #teste
                    # teste_dc = np.zeros(385 * 385, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_dc, self.domination_counter_g)
                    # teste_dc.shape = 385, 385
                    # teste_dict[self.generation_count]['dc32'] = teste_dc

                    # teste = np.zeros((256*3+1)*(256*3+1), dtype=np.int32)
                    # cuda.memcpy_dtoh(teste, self.domination_counter_g)
                    # teste.shape = 256*3+1, 256*3+1
                    #
                    # teste_f = np.zeros(256 * 3 * 3, np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 256 * 3, 3
                    # pass

                    fast_nondominated_sort3 = self.mod.get_function("fast_nondominated_sort3")
                    fast_nondominated_sort3(self.domination_counter_g, temp, temp, self.fronts_g, self.tams_fronts_g,
                                            self.rank_g,
                                            block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # fast_nondominated_sort3 = self.mod.get_function("fast_nondominated_sort3_teste")
                    # fast_nondominated_sort3(self.domination_counter_g, temp, temp, self.fronts_g, self.tams_fronts_g,
                    #                         self.rank_g, self.fitness_g,
                    #                         block=(1, 1, 1), grid=(1, 1, 1))
                    # cuda.Context.synchronize()

                    # fast_nondominated_sort3 = self.mod.get_function("fast_nondominated_sort3")
                    # fast_nondominated_sort3(self.domination_counter_g, self.params.population_size_g,
                    #                         self.params.population_size_g, self.fronts_g, self.tams_fronts_g,
                    #                         self.rank_g,
                    #                         block=(1, 1, 1), grid=(1, 1, 1))
                    # cuda.Context.synchronize()

                    # teste_f = np.zeros(256 * 3 * 3, np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 256 * 3, 3
                    # pass

                    # teste
                    # teste_front = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_front, self.fronts_g)
                    # teste_dict[self.generation_count]['f31'] = teste_front
                    # tam_front = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(tam_front, self.tams_fronts_g)
                    # teste_dict[self.generation_count]['tf31'] = tam_front
                    # pass

                    # teste2 = np.zeros(128, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste2, self.fronts_g)
                    # teste = np.zeros(261 * 30, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste, self.aux3_g)
                    # teste.shape = 261, 3, 10
                    # teste3 = np.zeros(261 * 30, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste3, self.personal_best_position_g)
                    # teste3.shape = 261, 3, 10
                    # pass

                gpu[6] += (dt.now() - start).total_seconds()

                # teste fronts
                # f = []
                # tam = []
                # for i in self.fronts:
                #     tam.append(0)
                #     f2 = []
                #     for j in i:
                #         tam[-1] += 1
                #         f2.append(np.where(np.array(self.population) == j))
                #         if len(f2[-1][0]) == 0:
                #             f2[-1] = (np.where(np.array(self.population_copy) == j))
                #             f2[-1] = f2[-1][0][0] + 128
                #         else:
                #             f2[-1] = f2[-1][0][0]
                #     f2.sort()
                #     f.extend(f2)
                #
                # teste_f = np.zeros(256, dtype=np.int32)
                # cuda.memcpy_dtoh(teste_f, self.fronts_g)
                # teste_tam = np.zeros(256, dtype=np.int32)
                # cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                # # print('fronts gpu', '\n', teste_f)
                # # print('fronts cpu', '\n', f)
                # print('diff\n', teste_f-f)
                # if self.generation_count == 7:
                #     pass

                # teste
                # print("teste antes pb geracao")
                # teste = np.zeros(261 * 30, dtype=np.float64)
                # cuda.memcpy_dtoh(teste, self.personal_best_position_g)
                # teste.shape = 261, 3, 10
                # teste2 = np.zeros(261 * 6, dtype=np.float64)
                # cuda.memcpy_dtoh(teste2, self.personal_best_fitness_g)
                # teste2.shape = 261, 3, 2
                #
                # l = []
                # l3 = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(abs(np.array(self.population[i].personal_best[j].position) - teste[i][k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(abs(np.array(self.population[i].personal_best[j].fitness) - teste2[i][k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # l3 = []
                # for i in range(len(self.population_copy)):
                #     for j in range(len(self.population_copy[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(
                #                     abs(np.array(self.population_copy[i].personal_best[j].position) - teste[i + 128][
                #                         k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # for i in range(len(self.population_copy)):
                #     for j in range(len(self.population_copy[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(
                #                     abs(np.array(self.population_copy[i].personal_best[j].fitness) - teste2[i + 128][
                #                         k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)

                ## Seleciona para a proxima geracao 0:00:00.010896 de 0:01:07.989052 0,0147%
                # se nao tiver copia, nao ha seleção pra proxima geracao?
                fronts3 = []
                fronts4 = []
                broken = False

                start = dt.now()

                # if self.copy_pop:
                #     next_generation = []
                #     i = 0
                #     while len(next_generation) < self.params.population_size:
                #         # #teste
                #         f2 = []
                #         if len(self.fronts[i]) + len(next_generation) <= self.params.population_size:
                #             for p in self.fronts[i]:
                #                 next_generation.append(p)
                #
                #                 # teste
                #                 if len(np.where(np.array(self.population) == p)[0]) > 0:
                #                     f2.append(np.where(np.array(self.population) == p)[0][0])
                #                 elif len(np.where(np.array(self.population_copy) == p)[0]) > 0:
                #                     f2.append(np.where(np.array(self.population_copy) == p)[0][0])
                #                     f2[-1] += 128
                #                 else:
                #                     f2.append(-1)
                #
                #             # #teste
                #             fronts3.append(f2)
                #             fronts4.extend(f2)
                #
                #             # teste
                #             # l = []
                #             # for k in next_generation:
                #             #     l.append(np.where(np.array(self.population) == k)[0])
                #             #     if (len(l[-1]) == 0):
                #             #         l[-1] = np.where(np.array(self.population_copy) == k)[0] + 128
                #             #     l[-1] = l[-1][0]
                #             # l.sort()
                #             # l = set(l)
                #
                #         else:
                #
                #             # teste
                #             # fronts = []
                #             # for k in self.fronts:
                #             #     f2 = []
                #             #     for j in k:
                #             #         if len(np.where(np.array(self.population) == j)[0]) > 0:
                #             #             f2.append(np.where(np.array(self.population) == j)[0][0])
                #             #         elif len(np.where(np.array(self.population_copy) == j)[0]) > 0:
                #             #             f2.append(np.where(np.array(self.population_copy) == j)[0][0])
                #             #             f2[-1] += 128
                #             #         else:
                #             #             f2.append(-1)
                #             #     fronts.append(f2)
                #             # print(fronts)
                #             # pass
                #
                #             # apagar apos validacao
                #             broken = True
                #
                #             if self.params.crowd_distance_type == 0:
                #                 self.crowding_distance(self.fronts[i])
                #             else:
                #                 # apenas para valkidacao, apagar depois
                #                 # a ordem das particulas para o calculo do crowding distance
                #                 # estava estranha quando muitas particulas tinham o mesmo fitness numa dimensao
                #                 # para afzer os resultados da cpu testeTempo250624 gpu iguais, copiei a ordem da cpu
                #                 # para validacao, depois apagar isso
                #                 # self.population_copy2 = copy.deepcopy(self.population)
                #                 # self.population_copy3 = copy.deepcopy(self.population_copy)
                #
                #                 # teste - o codigo da cpu nao estava zerando o crowd-distance
                #                 # das copias, testeTempo250624 isso talvez interfira no crowd_distance. Aqui etstei zerar
                #                 # todos os cd das copias para testar. Talvez apagar depois
                #                 for pc in self.population_copy:
                #                     pc.crowd_distance = 0
                #
                #                 # o codigo de cpu nao calcula o cd das copias, entao elas sempre sao zero.
                #                 # modifiquei para calcular tanto das particulas quanto das copias
                #                 # self.crowding_distance(self.population)
                #                 new_p = []
                #                 new_p.extend(self.population)
                #                 new_p.extend(self.population_copy)
                #                 self.crowding_distance(new_p)
                #
                #             # teste
                #             # fronts = []
                #             # for k in self.fronts:
                #             #     f2 = []
                #             #     for j in k:
                #             #         if len(np.where(np.array(self.population) == j)[0]) > 0:
                #             #             f2.append(np.where(np.array(self.population) == j)[0][0])
                #             #         elif len(np.where(np.array(self.population_copy) == j)[0]) > 0:
                #             #             f2.append(np.where(np.array(self.population_copy) == j)[0][0])
                #             #             f2[-1] += 128
                #             #         else:
                #             #             f2.append(-1)
                #             #     fronts.append(f2)
                #             # print(fronts)
                #             # pass
                #
                #             # teste
                #             # f2 = []
                #             # for j in self.fronts[i]:
                #             #     if len(np.where(np.array(self.population) == j)[0]) > 0:
                #             #         f2.append(np.where(np.array(self.population) == j)[0][0])
                #             #     elif len(np.where(np.array(self.population_copy) == j)[0]) > 0:
                #             #         f2.append(np.where(np.array(self.population_copy) == j)[0][0])
                #             #         f2[-1] += 128
                #             #     else:
                #             #         f2.append(-1)
                #             # pass
                #
                #             self.fronts[i].sort(key=lambda x: x.crowd_distance)
                #             j = len(self.fronts[i]) - 1
                #
                #             # teste
                #             # f2 = []
                #             # for k in self.fronts[i]:
                #             #     if len(np.where(np.array(self.population) == k)[0]) > 0:
                #             #         f2.append(np.where(np.array(self.population) == k)[0][0])
                #             #     elif len(np.where(np.array(self.population_copy) == k)[0]) > 0:
                #             #         f2.append(np.where(np.array(self.population_copy) == k)[0][0])
                #             #         f2[-1] += 128
                #             #     else:
                #             #         f2.append(-1)
                #             # pass
                #
                #             # teste
                #             f2 = []
                #
                #             while len(next_generation) < self.params.population_size:
                #                 next_generation.append(self.fronts[i][j])
                #                 # print(self.fronts[i][j].position)
                #                 # print(self.fronts[i][j].fitness)
                #
                #                 # teste
                #                 if len(np.where(np.array(self.population) == self.fronts[i][j])[0]) > 0:
                #                     f2.append(np.where(np.array(self.population) == self.fronts[i][j])[0][0])
                #                 elif len(np.where(np.array(self.population_copy) == self.fronts[i][j])[0]) > 0:
                #                     f2.append(np.where(np.array(self.population_copy) == self.fronts[i][j])[0][0])
                #                     f2[-1] += 128
                #                 else:
                #                     f2.append(-1)
                #                 j = j - 1
                #
                #             # teste
                #             fronts3.append(f2)
                #             fronts4.extend(f2)
                #
                #         i = i + 1
                #     self.population = next_generation

                    # teste
                    # l2 = []
                    # for i in next_generation:
                    #     l2.append(np.where(np.array(self.population) == i)[0])
                    #     if (len(l2[-1]) == 0):
                    #         l2[-1] = np.where(np.array(self.population_copy) == i)[0] + 128
                    #     l2[-1] = l2[-1][0]
                    # l2.sort()

                cpu[7] += (dt.now() - start).total_seconds()

                # teste = np.zeros(261 * 2, dtype=np.float64)
                # cuda.memcpy_dtoh(teste, self.fitness_g)
                # teste.shape = 261, 2
                # l = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].fitness)):
                #         l.append(abs(self.population[i].fitness[j] - teste[i][j]))
                # for i in range(len(self.population_copy)):
                #     for j in range(len(self.population_copy[i].fitness)):
                #         l.append(abs(self.population_copy[i].fitness[j] - teste[i + 128][j]))
                # l = np.array(l)
                # l = np.where(l > 1e-5)[0]
                # print('teste_fit', l)
                # pass

                # teste2 = np.zeros(128, dtype=np.int32)
                # cuda.memcpy_dtoh(teste2, self.fronts_g)
                # teste = np.zeros(261 * 30, dtype=np.float64)
                # cuda.memcpy_dtoh(teste, self.aux3_g)
                # teste.shape = 261, 3, 10
                # teste3 = np.zeros(261 * 30, dtype=np.float64)
                # cuda.memcpy_dtoh(teste3, self.personal_best_position_g)
                # teste3.shape = 261, 3, 10

                start = dt.now()

                # # teste
                # teste_cur = np.zeros(1, np.int32)
                # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                # teste_f = np.zeros(256*3 * 3, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 256*3, 3
                # teste_p = np.zeros(384 * 10, np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 384, 10
                # # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                # # plt.show()
                # pass

                if self.gpu:
                    nextgen1 = self.mod.get_function("nextgen1")
                    nextgen1(self.fronts_g, self.tams_fronts_g, self.params.population_size_g,
                             block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # teste next_gen1
                    # print('teste next_gen1')
                    # teste_tam = np.zeros(128, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                    # # a ideia foi aproveitar o espaço de tamanhos de fronts, ja que dificilmente
                    # # teremos um numeor de fornts igual ao numerod testeTempo250624 particulas
                    # print('numero de fronts cujo total testeTempo250624 menor que a populacao', teste_tam[-2])
                    # print('numero de particulas do ultimo front qe faltam para completar a populacao',
                    #       teste_tam[-1])

                    # teste_tam = np.zeros(128, dtype=np.int32)
                    teste_tam = np.zeros(self.params.population_size, dtype=np.int32)
                    cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                    if teste_tam[-1] > 0:
                        # zerar vetor crowding distance
                        crowding_distance_inicialization = self.mod.get_function("crowding_distance_inicialization")
                        crowding_distance_inicialization(self.crowding_distance_g,
                                                         block=(2*self.params.population_size, 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        # teste
                        # teste_cd = np.zeros(256, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste_cd, self.crowding_distance_g)
                        # print(teste_cd)

                        population_index_inicialization = self.mod.get_function("population_index_inicialization")
                        population_index_inicialization(self.population_index_g,
                                                        block=(2*self.params.population_size, 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        #teste
                        # teste = np.zeros(256, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste, self.population_index_g)
                        # print(teste)

                        # ordena o vetor de crowding distance testeTempo250624 population_index para auxiliar
                        # no calculo ddo crowding distance.
                        for i in range(self.params.objectives_dim):

                            # para validacao, apagar depois
                            # self.aux4.shape = 2, 256
                            # cuda.memcpy_htod(self.population_index_g, self.aux4[i])
                            # self.aux4 = self.aux4.flatten()

                            cuda.memcpy_htod(i_g, np.array([i], dtype=np.int32))

                            # para validacao, essa parte sera comentada, descomentar depois
                            # ordena os fronts em ordem crescente de cada coordenada fitness
                            # cuda.memcpy_htod(i_g, np.array([i], dtype=np.int32))
                            for j in range(self.params.population_size):
                                front_sort5_par = self.mod.get_function("front_sort5_par")
                                front_sort5_par(self.fitness_g,
                                                self.params.objectives_dim_g, i_g, self.crowding_distance_g,
                                                self.params.population_size_g, self.population_index_g,
                                                block=(int(self.params.population_size / 2), 1, 1), grid=(1, 1, 1))
                                cuda.Context.synchronize()

                                # teste = np.zeros(256, dtype=np.int32)
                                # cuda.memcpy_dtoh(teste, self.population_index_g)
                                # print(teste[:128])
                                # pass

                                # cuda.memcpy_htod(i_g, np.array([i], dtype=np.int32))
                                front_sort5_impar = self.mod.get_function("front_sort5_impar")
                                front_sort5_impar(self.fitness_g,
                                                  self.params.objectives_dim_g, i_g, self.crowding_distance_g,
                                                  self.params.population_size_g, self.population_index_g,
                                                  block=(int(self.params.population_size / 2) - 1, 1, 1), grid=(1, 1, 1))
                                cuda.Context.synchronize()

                            # tam_front = np.zeros(self.params.population_size, dtype=np.int32)
                            # cuda.memcpy_dtoh(tam_front, self.tams_fronts_g)
                            crowding_distance4 = self.mod.get_function("crowding_distance4")
                            crowding_distance4(self.fitness_g,
                                               self.params.objectives_dim_g, self.tams_fronts_g, i_g,
                                               self.crowding_distance_g, self.params.population_size_g,
                                               self.population_index_g,
                                               block=(int(2*self.params.population_size) - 2, 1, 1), grid=(1, 1, 1))

                            #teste
                            # teste_f = np.zeros(261 * 2, dtype=np.float64)
                            # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                            # teste_f.shape = 261, 2
                            # teste_pi = np.zeros(256, dtype=np.int32)
                            # cuda.memcpy_dtoh(teste_pi, self.population_index_g)
                            # # print(teste_f[teste_pi[0:20]])
                            # teste_fr = np.zeros(256, dtype=np.int32)
                            # cuda.memcpy_dtoh(teste_fr, self.fronts_g)
                            # teste_cd = np.zeros(self.params.population_size*2, dtype=np.float64)
                            # cuda.memcpy_dtoh(teste_cd, self.crowding_distance_g)
                            # print(teste_cd[:12])
                            # pass

                        # teste crowding distance
                        # teste_cd = np.zeros(256, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_cd, self.crowding_distance_g)

                        # teste population_index
                        # teste_pi = np.zeros(256, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste_pi, self.population_index_g)
                        # print('population_index antes')
                        # print(teste_pi[:128])

                        for j in range(self.params.population_size):
                            index_sort_par = self.mod.get_function("index_sort_par")
                            index_sort_par(self.crowding_distance_g, self.population_index_g,
                                           block=(int(self.params.population_size / 2), 1, 1), grid=(1, 1, 1))
                            cuda.Context.synchronize()

                            # teste = np.zeros(256, dtype=np.int32)
                            # cuda.memcpy_dtoh(teste, self.population_index_g)
                            # print(teste[:128])
                            # pass

                            index_sort_impar = self.mod.get_function("index_sort_impar")
                            index_sort_impar(self.crowding_distance_g, self.population_index_g,
                                             block=(int(self.params.population_size / 2) - 1, 1, 1), grid=(1, 1, 1))
                            cuda.Context.synchronize()

                            # teste = np.zeros(256, dtype=np.int32)
                            # cuda.memcpy_dtoh(teste, self.population_index_g)
                            # print(teste[:128])
                            # pass

                        # teste population_index
                        # teste_pi = np.zeros(256, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste_pi, self.population_index_g)
                        # print('population_index depois')
                        # print(teste_pi[:128])

                        #teste
                        # teste_f = np.zeros(256, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste_f, self.fronts_g)
                        # teste_tam = np.zeros(256, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                        # teste_cd = np.zeros(self.params.population_size * 2, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_cd, self.crowding_distance_g)
                        # pass

                        front_sort_crowding_distance4 = self.mod.get_function("front_sort_crowding_distance4")
                        front_sort_crowding_distance4(self.fronts_g, self.tams_fronts_g,
                                                      self.crowding_distance_g, self.params.population_size_g,
                                                      block=(1, 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        # teste
                        # teste_f = np.zeros(256, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste_f, self.fronts_g)
                        # teste_tam = np.zeros(256, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                        # teste_cd = np.zeros(self.params.population_size * 2, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_cd, self.crowding_distance_g)
                        # pass



                #teste front quebrado
                # teste_p = np.zeros(261*10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 261, 10
                # teste_f = np.zeros(261 * 2, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 261, 2
                # teste_fr = np.zeros(256, dtype=np.int32)
                # cuda.memcpy_dtoh(teste_fr, self.fronts_g)
                # teste_tam = np.zeros(256, dtype=np.int32)
                # cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                # tam = len(fronts3[-1])
                # a1 = list(range(128-tam, 128))
                # a2 = teste_fr[128-tam:128]

                # print('teste position')
                # teste_p = np.zeros(261 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 261, 10
                # l2 = []
                # for i in range(len(a1)):
                #     for j in range(len(self.population[a1[i]].position)):
                #         l2.append(abs(self.population[a1[i]].position[j] - teste_p[a2[i]][j]))
                # l2 = np.where(np.array(l2) > 1e-5)[0]
                # print(l2)

                # if(len(l2) > 0):
                #     l3 = list(set(np.array(l2)//self.params.position_dim))
                # pass

                # create the next gen
                # comando encessario para validação. Depois apagar
                # necessario pq os fronts da cpu estao ordenados de forma diferente dos fronts da gpu
                fronts5 = []
                # broken = True
                # if broken:
                #     for i in fronts3[:-1]:
                #         fronts5.extend(i)
                # else:
                #     for i in fronts3:
                #         fronts5.extend(i)
                #
                # cuda.memcpy_htod(self.fronts_g, np.array(fronts5, dtype=np.int32))

                # teste position
                # print('teste antes position')

                # div = 2
                div = int(self.params.population_size/64)
                # copiar os selecionados para uma area auxiliar
                create_next_gen1 = self.mod.get_function("create_next_gen1")
                create_next_gen1(self.position_g, self.aux_g, self.fronts_g,
                                 block=(int(self.params.population_size / div), self.params.position_dim, 1),
                                 grid=(div, 1, 1))
                cuda.Context.synchronize()

                create_next_gen2 = self.mod.get_function("create_next_gen2")
                create_next_gen2(self.position_g, self.aux_g,
                                 block=(int(self.params.population_size / div), self.params.position_dim, 1),
                                 grid=(div, 1, 1))
                cuda.Context.synchronize()

                # teste position
                # print('teste depois position')
                # teste_p = np.zeros(261 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 261, 10
                # l2 = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].position)):
                #         l2.append(abs(self.population[i].position[j] - teste_p[i][j]))
                # l2 = np.array(l2)
                # l2 = np.where(abs(l2) > 1e-5)[0]
                # print(l2)
                # pass

                # teste velocity
                # print('teste antes velocity')

                create_next_gen1 = self.mod.get_function("create_next_gen1")
                create_next_gen1(self.velocity_g, self.aux_g, self.fronts_g,
                                 block=(int(self.params.population_size / div), self.params.position_dim, 1),
                                 grid=(div, 1, 1))
                cuda.Context.synchronize()

                create_next_gen2 = self.mod.get_function("create_next_gen2")
                create_next_gen2(self.velocity_g, self.aux_g,
                                 block=(int(self.params.population_size / div), self.params.position_dim, 1),
                                 grid=(div, 1, 1))
                cuda.Context.synchronize()

                # teste velocity
                # print('teste depois velocity')
                # teste_v = np.zeros(261 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_v, self.velocity_g)
                # teste_v.shape = 261, 10
                # l2 = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].velocity)):
                #         l2.append(abs(self.population[i].velocity[j] - teste_v[i][j]))
                # l2 = np.array(l2)
                # l2 = np.where(abs(l2) > 1e-5)[0]
                # print('teste vel', l2)

                # div = 8
                div = int(self.params.population_size/16)
                create_next_gen1 = self.mod.get_function("create_next_gen1")
                create_next_gen1(self.personal_best_position_g, self.aux3_g, self.fronts_g,
                                 block=(int(self.params.population_size / div),
                                        self.params.position_dim * self.params.personal_guide_array_size, 1),
                                 grid=(div, 1, 1))
                cuda.Context.synchronize()

                create_next_gen2 = self.mod.get_function("create_next_gen2")
                create_next_gen2(self.personal_best_position_g, self.aux3_g,
                                 block=(int(self.params.population_size / div),
                                        self.params.position_dim * self.params.personal_guide_array_size, 1),
                                 grid=(div, 1, 1))
                cuda.Context.synchronize()

                # div = 2
                div = int(self.params.population_size/64)
                create_next_gen1 = self.mod.get_function("create_next_gen1")
                create_next_gen1(self.personal_best_fitness_g, self.aux3_g, self.fronts_g,
                                 block=(int(self.params.population_size / div),
                                        self.params.objectives_dim * self.params.personal_guide_array_size, 1),
                                 grid=(div, 1, 1))
                cuda.Context.synchronize()

                create_next_gen2 = self.mod.get_function("create_next_gen2")
                create_next_gen2(self.personal_best_fitness_g, self.aux3_g,
                                 block=(int(self.params.population_size / div),
                                        self.params.objectives_dim * self.params.personal_guide_array_size, 1),
                                 grid=(div, 1, 1))
                cuda.Context.synchronize()

                gpu[7] += (dt.now() - start).total_seconds()

                #teste
                # teste_cur = np.zeros(1, np.int32)
                # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                # teste_f = np.zeros(384 * 2, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 384, 2
                # teste_p = np.zeros(384 * 10, np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 384, 10
                # # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                # # plt.show()
                # pass

                # teste
                # print("teste depois pb")
                # teste = np.zeros(384 * 30, dtype=np.float64)
                # cuda.memcpy_dtoh(teste, self.personal_best_position_g)
                # teste.shape = 384, 3, 10
                # teste2 = np.zeros(384 * 6, dtype=np.float64)
                # cuda.memcpy_dtoh(teste2, self.personal_best_fitness_g)
                # teste2.shape = 384, 3, 2
                # teste3 = np.zeros(384 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste3, self.position_g)
                # teste3.shape = 384, 10
                # teste4 = np.zeros(384 * 2, dtype=np.float64)
                # cuda.memcpy_dtoh(teste4, self.fitness_g)
                # teste4.shape = 384, 2
                # pass
                #
                # l = []
                # l3 = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(abs(np.array(self.population[i].personal_best[j].position) - teste[i][k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(abs(np.array(self.population[i].personal_best[j].fitness) - teste2[i][k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # l3 = []
                # for i in range(len(self.population_copy)):
                #     for j in range(len(self.population_copy[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(
                #                     abs(np.array(self.population_copy[i].personal_best[j].position) - teste[i + 128][
                #                         k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)
                #
                # l = []
                # for i in range(len(self.population_copy)):
                #     for j in range(len(self.population_copy[i].personal_best)):
                #         l2 = []
                #         for k in range(3):
                #             l2.append(
                #                 list(
                #                     abs(np.array(self.population_copy[i].personal_best[j].fitness) - teste2[i + 128][
                #                         k])))
                #         l2.sort()
                #         l.extend(l2[0])
                #         l3.append((i, j))
                # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                # print(l2)

                # ordenar positions para validacao, apgar depois
                # verificar essa parte depois pois esta estranha
                # l = np.zeros(10, dtype=np.int32)
                # l = np.array(l, dtype=np.int32)
                # l_g = cuda.mem_alloc(l.nbytes)
                # cuda.memcpy_htod(l_g, l)
                # for j in range(self.params.population_size):
                #     index_sort_par2 = self.mod.get_function("index_sort_par2")
                #     index_sort_par2(l_g, self.position_g, self.velocity_g, self.params.position_dim_g,
                #                     block=(int(self.params.population_size / 2), 1, 1), grid=(1, 1, 1))
                #     cuda.Context.synchronize()
                #
                #     # teste = np.zeros(261 * 10, dtype=np.float64)
                #     # cuda.memcpy_dtoh(teste, self.position_g)
                #     # teste.shape = 261, 10
                #     # teste2 = np.zeros(128, dtype=np.int32)
                #     # cuda.memcpy_dtoh(teste2, l_g)
                #
                #     index_sort_impar2 = self.mod.get_function("index_sort_impar2")
                #     index_sort_impar2(l_g, self.position_g, self.velocity_g, self.params.position_dim_g,
                #                       block=(int(self.params.population_size / 2) - 1, 1, 1), grid=(1, 1, 1))
                #     cuda.Context.synchronize()

                # teste position
                # print('teste position')
                # teste_p = np.zeros(261 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 261, 10
                # l2 = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].position)):
                #         l2.append(abs(self.population[i].position[j] - teste_p[i][j]))
                # l2 = np.array(l2)
                # l2 = np.where(abs(l2) > 1e-5)[0]
                # print(l2)
                # pass
                #
                # # teste fitness
                # print('teste fitness')
                # teste_f = np.zeros(261 * 2, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 261, 2
                # l2 = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].position)):
                #         l2.append(abs(self.population[i].position[j] - teste_p[i][j]))
                # l2 = np.array(l2)
                # l2 = np.where(abs(l2) > 1e-5)[0]
                # print(l2)
                # pass
                #
                # # teste velocity
                # print('teste velocity')
                # teste_v = np.zeros(261 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_v, self.velocity_g)
                # teste_v.shape = 261, 10
                # l2 = []
                # for i in range(len(self.population)):
                #     for j in range(len(self.population[i].velocity)):
                #         l2.append(abs(self.population[i].velocity[j] - teste_v[i][j]))
                # l2 = np.array(l2)
                # l2 = np.where(abs(l2) > 1e-5)[0]
                # print(l2)
                # if self.generation_count == 6:
                #     pass
                #
                # # teste memoria
                # print('teste memoria antes')
                # teste_p = np.zeros(261 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 261, 10
                # l2 = []
                # for i in range(len(self.memory)):
                #     for j in range(len(self.memory[i].position)):
                #         l2.append(abs(self.memory[i].position[j] - teste_p[i + 256][j]))
                # l2 = np.array(l2)
                # l2 = np.where(abs(l2) > 1e-5)[0]
                # print(l2)
                # if self.generation_count == 7:
                #     pass


                # l = []
                # for i in self.population:
                #     l.append(i>>self.population[5])
                # l2 = []
                # for i in self.population:
                #     l2.append(i >> self.population[9])

                start = dt.now()

                # self.memory_update()  # 0:00:02.644841 de  0:00:44.628465

                cpu[8] += (dt.now() - start).total_seconds()

                # teste memoria
                # print('teste memoria antes gpu')

                start = dt.now()

                # f_teste = open('testeNAN2.pkl', 'wb')
                # dict2 = {}

                if self.gpu:
                    # cria uma lista de membros do front0 testeTempo250624 da memoria atual
                    inicialize_front0_mem2 = self.mod.get_function("inicialize_front0_mem2")
                    inicialize_front0_mem2(self.fronts_g, self.front0_mem_g, self.tams_fronts_g,
                                           self.tam_front0_mem_g, self.position_g, self.params.memory_size_g,
                                           self.params.population_size_g,
                                           self.params.position_dim_g, self.params.current_memory_size_g,
                                           block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # teste
                    # teste_cur = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                    # # teste_dict[self.generation_count]['cur41'] = teste_cur
                    # teste_fr = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_fr, self.fronts_g)
                    # teste_dict[self.generation_count]['f41'] = teste2
                    # teste_tam = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                    # # teste_dict[self.generation_count]['tf41'] = teste_tam
                    # teste3 = np.zeros(384, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste3, self.front0_mem_g)
                    # # teste_dict[self.generation_count]['fronts0_mem_41'] = teste3
                    # teste4 = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste4, self.tam_front0_mem_g)
                    # # teste_dict[self.generation_count]['tam_fronts0_mem_41'] = teste4
                    # teste_p = np.zeros(384 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = 384, 10
                    # # teste_dict[self.generation_count]['pos41'] = teste_p
                    # teste_f = np.zeros(384 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # teste_dict[self.generation_count]['fit41'] = teste_f

                    # teste nan
                    # teste_cur = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                    # dict2['cur'] = teste_cur
                    # teste2 = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste2, self.fronts_g)
                    # dict2['fronts'] = teste2
                    # teste_tam = np.zeros(128, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)
                    # dict2['tam_fronts'] = teste_tam
                    # teste3 = np.zeros(256, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste3, self.front0_mem_g)
                    # dict2['fronts0_mem'] = teste3
                    # teste4 = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste4, self.tam_front0_mem_g)
                    # dict2['tam_fronts0_mem'] = teste4
                    # teste_p = np.zeros(384 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = 384, 10
                    # dict2['pos'] = teste_p
                    # teste_f = np.zeros(384 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # dict2['fit'] = teste_f

                    # teste
                    # teste = np.zeros(128, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste, self.front0_mem_g)

                    # if func == 'ZDT1':
                    # zdt1 = self.mod.get_function("zdt1")
                    div = int(np.ceil((self.params.population_size * 2 + self.params.memory_size) / 512))
                    function = self.mod.get_function("function")
                    function(self.params.func_n_g, self.position_g, self.params.position_dim_g,
                             self.fitness_g, self.alpha_g,
                         block=(int((2 * self.params.population_size + self.params.memory_size)/div), 1, 1), grid=(div, 1, 1))
                    cuda.Context.synchronize()
                    self.fitness_eval_count += 2*self.params.population_size+self.params.memory_size

                    # teste_f = np.zeros(384 * 2, np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                    # plt.show()
                    # pass

                    #teste
                    # teste_f = np.zeros(384 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # teste_dict[self.generation_count]['fit42'] = teste_f

                    # teste position
                    # print('teste position')
                    # teste_p = np.zeros(261 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_p, self.position_g)
                    # teste_p.shape = 261, 10
                    # l2 = []
                    # for i in range(len(self.population)):
                    #     for j in range(len(self.population[i].position)):
                    #         l2.append(abs(self.population[i].position[j] - teste_p[i][j]))
                    # l2 = np.array(l2)
                    # l2 = np.where(abs(l2) > 1e-5)[0]
                    # print(l2)
                    # pass
                    #
                    # # teste fitness
                    # print('teste fitness')
                    # teste_f = np.zeros(261 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 261, 2
                    # l2 = []
                    # for i in range(len(self.population)):
                    #     for j in range(len(self.population[i].position)):
                    #         l2.append(abs(self.population[i].position[j] - teste_p[i][j]))
                    # l2 = np.array(l2)
                    # l2 = np.where(abs(l2) > 1e-5)[0]
                    # print(l2)
                    # pass

                    tam_front = np.zeros(1, dtype=np.int32)
                    cuda.memcpy_dtoh(tam_front, self.tam_front0_mem_g)
                    if tam_front>32:
                        block_x = 32
                        grid_x = int(np.ceil(tam_front[0]/32))
                    else:
                        block_x = int(tam_front[0])
                        grid_x = 1
                    # print('gx', grid_x, 'bx', block_x)

                    # fast_nondominated_sort4 = self.mod.get_function("fast_nondominated_sort4")
                    # fast_nondominated_sort4(self.fitness_g, self.params.objectives_dim_g,
                    #                         self.domination_counter_g, self.params.population_size_g,
                    #                         self.params.otimizations_type_g, self.params.objectives_dim_g,
                    #                         self.front0_mem_g, self.tam_front0_mem_g,
                    #                         # block=(int(tam_front[0]), int(tam_front[0]), 1),
                    #                         # grid=(1, 1, 1))
                    #                         block=(block_x, block_x, 1),
                    #                         grid=(grid_x, grid_x, 1))
                    # cuda.Context.synchronize()
                    fast_nondominated_sort4_2 = self.mod.get_function("fast_nondominated_sort4_2")
                    fast_nondominated_sort4_2(self.fitness_g, self.params.objectives_dim_g,
                                            self.domination_counter_g, self.params.population_size_g,
                                            self.params.otimizations_type_g, self.params.objectives_dim_g,
                                            self.front0_mem_g, self.tam_front0_mem_g,
                                            # block=(int(tam_front[0]), int(tam_front[0]), 1),
                                            # grid=(1, 1, 1))
                                            block=(block_x, block_x, 1),
                                            grid=(grid_x, grid_x, 1))
                    cuda.Context.synchronize()

                    # teste4 = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste4, self.tam_front0_mem_g)
                    # print('tam front0mesm python', teste4)

                    # teste
                    # teste_f = np.zeros(384 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                    # teste_f.shape = 384, 2
                    # teste_dict[self.generation_count]['fit43'] = teste_f
                    # teste_d = np.zeros(385 * 385, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste_d, self.domination_counter_g)
                    # teste_d.shape = 385, 385
                    # teste_dict[self.generation_count]['dc42'] = teste_d
                    # teste3 = np.zeros(384, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste3, self.front0_mem_g)
                    # teste_dict[self.generation_count]['fronts0_mem_42'] = teste3
                    # teste4 = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste4, self.tam_front0_mem_g)
                    # teste_dict[self.generation_count]['tam_fronts0_mem_42'] = teste4

                    # teste2 = np.zeros(385 * 385, dtype=np.int32)
                    # cuda.memcpy_dtoh(teste2, self.domination_counter_g)
                    # teste2.shape = 385, 385
                    # teste3 = np.zeros(261 * 2, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste3, self.fitness_g)
                    # teste3.shape = 261, 2
                    # teste4 = np.zeros(261 * 10, dtype=np.float64)
                    # cuda.memcpy_dtoh(teste4, self.position_g)
                    # teste4.shape = 261, 10

                    fast_nondominated_sort5 = self.mod.get_function("fast_nondominated_sort5")
                    fast_nondominated_sort5(self.domination_counter_g,
                                            block=(int(tam_front[0]), 1, 1),
                                            grid=(1, 1, 1))
                    cuda.Context.synchronize()

                    # tam_front = np.zeros(1, dtype=np.int32)
                    # cuda.memcpy_dtoh(tam_front, self.tam_front0_g)
                    # print(int(tam_front[0]), tam_front[0])

                    fast_nondominated_sort6 = self.mod.get_function("fast_nondominated_sort6")
                    fast_nondominated_sort6(self.domination_counter_g, self.tam_front0_mem_g,
                                            self.front0_mem_g, self.tam_front0_g, self.front0_g,
                                            block=(1, 1, 1), grid=(1, 1, 1))
                    cuda.Context.synchronize()



                    # teste
                    # print(int(tam_front[0]), tam_front[0])

                    tam_front = np.zeros(1, dtype=np.int32)
                    cuda.memcpy_dtoh(tam_front, self.tam_front0_g)
                    # atualiza memoria pela GPU
                    if tam_front[0] <= self.params.memory_size:
                        teste1 = 2
                        cuda.memcpy_htod(self.params.current_memory_size_g, tam_front[0])
                        copy3 = self.mod.get_function("copy3")
                        copy3(self.position_g, self.aux_g, self.params.position_dim_g,
                              block=(int(2 * self.params.population_size + self.params.memory_size), 1, 1),
                              grid=(1, 1, 1))
                        cuda.Context.synchronize()
                        copy3(self.velocity_g, self.aux2_g, self.params.position_dim_g,
                              block=(int(2 * self.params.population_size + self.params.memory_size), 1, 1),
                              grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        #teste nan
                        # teste_p = np.zeros(384 * 10, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_p, self.position_g)
                        # teste_p.shape = 384, 10
                        # dict2['pos3'] = teste_p

                        # teste2 = np.zeros(261 * 10, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste2, self.aux2_g)
                        # teste2.shape = 261, 10

                        memory_inicialization4 = self.mod.get_function("memory_inicialization4")
                        memory_inicialization4(self.position_g, self.fitness_g,
                                               self.params.position_dim_g, self.params.objectives_dim_g,
                                               self.params.population_size_g, self.aux_g,
                                               block=(int(self.params.memory_size), 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        #teste nan
                        # teste_p = np.zeros(384 * 10, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_p, self.position_g)
                        # teste_p.shape = 384, 10
                        # dict2['pos4'] = teste_p

                        # precisa ver depois se eralmente nao precisa da velocidade das particulas da memoria,
                        # mas por enquanto funcionou. O programa por enquanto copia errado a velocidade das
                        # particulas da memoria, ja que na fncao de movimento so calcula a velocidade dapopulação testeTempo250624 sua copia
                        # memory_inicialization5 = self.mod.get_function("memory_inicialization5")
                        # memory_inicialization5(self.position_g, self.velocity_g, self.fitness_g,
                        #                        self.aux_g, self.aux2_g, self.front0_g,
                        #                        self.params.position_dim_g, self.params.objectives_dim_g,
                        #                        self.params.population_size_g,
                        #                        block=(int(self.params.memory_size), 1, 1), grid=(1, 1, 1))
                        # cuda.Context.synchronize()
                        # print(int(tam_front[0]), tam_front[0])

                        # if tam_front[0] == -1:
                        #     pass

                        memory_inicialization5 = self.mod.get_function("memory_inicialization5")
                        memory_inicialization5(self.position_g, self.velocity_g, self.fitness_g,
                                               self.aux_g, self.aux2_g, self.front0_g,
                                               self.params.position_dim_g, self.params.objectives_dim_g,
                                               self.params.population_size_g,
                                               block=(int(tam_front[0]), 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        #teste nan
                        # teste_p = np.zeros(384 * 10, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_p, self.position_g)
                        # teste_p.shape = 384, 10
                        # dict2['pos5'] = teste_p
                        # teste = np.zeros(128, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste, self.front0_g)
                        # dict2['front0_2'] = teste
                        # teste = np.zeros(1, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste, self.tam_front0_g)
                        # dict2['tam_front0_2'] = teste
                        #
                        # pickle.dump(dict2, f_teste)
                        # f_teste.close()

                        # teste_f = np.zeros(384 * 2, np.float64)
                        # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                        # teste_f.shape = 384, 2
                        # plt.plot(teste_f[:256, 0], teste_f[:256, 1], 'ro')
                        # plt.show()
                        pass

                        # atualizar fitness da memoria nova
                        # if func == 'ZDT1':
                        #     zdt1 = self.mod.get_function("zdt1")
                        function = self.mod.get_function("function")
                        function(self.params.func_n_g, self.position_g, self.params.position_dim_g,
                                 self.fitness_g, self.alpha_g,
                             block=(2 * self.params.population_size + self.params.memory_size, 1, 1),
                             grid=(1, 1, 1))
                        cuda.Context.synchronize()
                        self.fitness_eval_count += 2*self.params.population_size+self.params.memory_size

                        # teste
                        # teste_f = np.zeros(384 * 2, np.float64)
                        # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                        # teste_f.shape = 384, 2
                        # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                        # plt.show()
                        pass

                        # teste_f = np.zeros(384 * 2, np.float64)
                        # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                        # teste_f.shape = 384, 2
                        # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                        # plt.show()
                        # pass

                        # teste
                        # teste = np.zeros(261 * 10, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste, self.position_g)
                        # teste.shape = 261, 10
                        # l3 = []
                        # for i in range(len(self.memory)):
                        #     for j in range(len(self.memory[i].position)):
                        #         l3.append(abs(self.memory[i].position[j] - teste[i + 256][j]))
                        # l3 = np.array(l3)
                        # l3 = np.where(l3 > 1e-5)[0]
                        # print('teste_gen', l3)

                        # teste2 = np.zeros(261 * 10, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste2, self.velocity_g)
                        # teste2.shape = 261, 10
                        # l4 = []
                        # for i in range(len(self.memory)):
                        #     for j in range(len(self.memory[i].velocity)):
                        #         l4.append(abs(self.memory[i].velocity[j] - teste2[i + 256][j]))
                        # l4 = np.array(l4)
                        # l4 = np.where(l4 > 1e-5)[0]
                        # print('teste_v', l4)
                        # pass

                    else:
                        # teste1 = 3
                        # falta testar o else quando o exemplo chegar aqui

                        # zerar vetor crowding distance
                        crowding_distance_inicialization = self.mod.get_function("crowding_distance_inicialization")
                        crowding_distance_inicialization(self.crowding_distance_g,
                                                         block=(self.params.population_size, 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        # teste
                        # teste_cd = np.zeros(384, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_cd, self.crowding_distance_g)
                        # teste_dict[self.generation_count]['cd_51'] = teste_cd

                        i_g = cuda.mem_alloc(np.array([1], np.int32).nbytes)
                        for i in range(self.params.objectives_dim):
                            # ordena os fronts em ordem crescente de cada coordenada fitness
                            cuda.memcpy_htod(i_g, np.array([i], dtype=np.int32))
                            front_sort = self.mod.get_function("front_sort")
                            front_sort(self.front0_g, self.tam_front0_g, self.fitness_g,
                                       self.params.objectives_dim_g, i_g, self.crowding_distance_g,
                                       block=(1, 1, 1), grid=(1, 1, 1))
                            cuda.Context.synchronize()

                            # teste
                            # teste2 = np.zeros(128, dtype=np.int32)
                            # cuda.memcpy_dtoh(teste2, self.front0_g)
                            # teste3 = np.zeros(261*2, dtype=np.float64)
                            # cuda.memcpy_dtoh(teste3, self.fitness_g)
                            # teste3.shape = 261, 2
                            # print(teste2[:15])
                            # print(teste3[:7])
                            # print(teste3[-5:])

                            crowding_distance = self.mod.get_function("crowding_distance")
                            crowding_distance(self.front0_g, self.tam_front0_g, self.fitness_g,
                                              self.params.objectives_dim_g, self.tams_fronts_g, i_g,
                                              self.crowding_distance_g,
                                              block=(int(tam_front[0] + self.params.memory_size - 2), 1, 1),
                                              grid=(1, 1, 1))

                            # teste
                            # teste4 = np.zeros(261, dtype=np.float64)
                            # cuda.memcpy_dtoh(teste4, self.crowding_distance_g)

                            pass

                        #teste
                        # teste_cd = np.zeros(384, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_cd, self.crowding_distance_g)
                        # teste_dict[self.generation_count]['cd_52'] = teste_cd

                        front_sort_crowding_distance = self.mod.get_function("front_sort_crowding_distance")
                        front_sort_crowding_distance(self.front0_g, self.tam_front0_g,
                                                     self.crowding_distance_g,
                                                     block=(1, 1, 1), grid=(1, 1, 1))

                        #teste
                        # teste3 = np.zeros(384, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste3, self.front0_g)
                        # teste_dict[self.generation_count]['fronts0_53'] = teste3
                        # teste4 = np.zeros(1, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste4, self.tam_front0_g)
                        # teste_dict[self.generation_count]['tam_fronts0_54'] = teste4

                        # teste
                        # teste_pos = np.zeros(261*10, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_pos, self.position_g)
                        # teste_pos.shape = 261, 10
                        # teste_fit = np.zeros(261 * 2, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste_fit, self.fitness_g)
                        # teste_fit.shape = 261, 2
                        # teste3 = np.zeros(261, dtype=np.float64)
                        # cuda.memcpy_dtoh(teste3, self.crowding_distance_g)
                        # print(teste3[:10])
                        # teste4 = np.zeros(self.params.population_size, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste4, self.front0_g)
                        # print(teste4[:10])
                        # pass

                        # talvez tenha que trocar a funcao

                        #testar se o problema e a inicializacao de memoria 190724
                        # memory_inicialization2 = self.mod.get_function("memory_inicialization2")
                        # memory_inicialization2(self.position_g, self.fitness_g, self.front0_g,
                        #                        self.params.position_dim_g, self.params.objectives_dim_g,
                        #                        self.params.population_size_g,
                        #                        block=(self.params.memory_size, 1, 1), grid=(1, 1, 1))
                        # cuda.Context.synchronize()

                        cuda.memcpy_htod(self.params.current_memory_size_g, tam_front[0])
                        copy3 = self.mod.get_function("copy3")
                        copy3(self.position_g, self.aux_g, self.params.position_dim_g,
                              block=(int(2 * self.params.population_size + self.params.memory_size), 1, 1),
                              grid=(1, 1, 1))
                        cuda.Context.synchronize()
                        copy3(self.velocity_g, self.aux2_g, self.params.position_dim_g,
                              block=(int(2 * self.params.population_size + self.params.memory_size), 1, 1),
                              grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        memory_inicialization4 = self.mod.get_function("memory_inicialization4")
                        memory_inicialization4(self.position_g, self.fitness_g,
                                               self.params.position_dim_g, self.params.objectives_dim_g,
                                               self.params.population_size_g, self.aux_g,
                                               block=(int(self.params.memory_size), 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        memory_inicialization5 = self.mod.get_function("memory_inicialization5")
                        memory_inicialization5(self.position_g, self.velocity_g, self.fitness_g,
                                               self.aux_g, self.aux2_g, self.front0_g,
                                               self.params.position_dim_g, self.params.objectives_dim_g,
                                               self.params.population_size_g,
                                               block=(int(self.params.memory_size), 1, 1), grid=(1, 1, 1))
                        cuda.Context.synchronize()

                        # atualizar fitness da memoria nova
                        # if func == 'ZDT1':
                        # zdt1 = self.mod.get_function("zdt1")
                        function = self.mod.get_function("function")
                        function(self.params.func_n_g, self.position_g, self.params.position_dim_g,
                                 self.fitness_g,
                             block=(2 * self.params.population_size + self.params.memory_size, 1, 1),
                             grid=(1, 1, 1))
                        cuda.Context.synchronize()
                        self.fitness_eval_count += 2*self.params.population_size +self.params.memory_size

                        cuda.memcpy_htod(self.params.current_memory_size_g,
                                         np.array([self.params.memory_size], dtype=np.int32))

                gpu[8] += (dt.now() - start).total_seconds()

                #teste
                # teste_f = np.zeros(384 * 2, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 384, 2
                # teste_dict[self.generation_count]['fit6'] = teste_f
                # teste_p = np.zeros(384 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 384, 10
                # teste_dict[self.generation_count]['pos6'] = teste_p
                # teste_cur = np.zeros(1, dtype=np.int32)
                # cuda.memcpy_dtoh(teste_cur, self.params.current_memory_size_g)
                # teste_dict[self.generation_count]['cur6'] = teste_cur

                # teste_nan
                # teste_f = np.zeros(384 * 2, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 384, 2
                # teste_p = np.zeros(384 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 384, 10
                # a = np.isnan(teste_f)
                # a = sum(sum(a))
                # b = teste_p<0
                # b = sum(sum(b))
                # if a > 0 or b>0:
                #     f_teste = open('testeNAN2.pkl', 'rb')
                #     d = pickle.load(f_teste)
                #     print(self.generation_count)
                #     a = np.where(np.isnan(teste_f) == True)
                #     exit(1)

                # teste memoria - necessario para deixar as simulacoes cpu testeTempo250624 gpu iguais
                # print('teste memoria depois gpu')
                # teste_p = np.zeros(261 * 10, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_p, self.position_g)
                # teste_p.shape = 261, 10
                # teste_f = np.zeros(261 * 2, dtype=np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 261, 2
                # l2 = []
                # for i in range(len(self.memory)):
                #     for j in range(len(self.memory[i].position)):
                #         l2.append(abs(self.memory[i].position[j] - teste_p[i + 256][j]))
                # l2 = np.array(l2)
                # l2 = np.where(abs(l2) > 1e-5)[0]
                # print(l2)
                # l5 = []
                # for i in range(len(self.memory)):
                #     for j in range(len(self.memory[i].fitness)):
                #         l5.append(abs(self.memory[i].fitness[j] - teste_f[i + 256][j]))
                # l5 = np.array(l5)
                # l5 = np.where(abs(l5) > 1e-5)[0]
                # print(l5)
                # if len(l2 > 0):
                #     l3 = list(set(l2 // 10))
                #     print(l3)
                #     if len(l3) == 2:
                #         l4 = []
                #         l4.extend(abs(self.memory[l3[0]].position - teste_p[l3[1] + 256]))
                #         l4.extend(abs(self.memory[l3[1]].position - teste_p[l3[0] + 256]))
                #         l4 = np.array(l4)
                #         l4 = np.where(abs(l4) > 1e-5)[0]
                #         print(l4)
                #         if len(l4) == 0:
                #             for j in range(len(self.memory[l3[0]].position)):
                #                 teste_p[256+l3[0]][j] = self.memory[l3[0]].position[j]
                #             for j in range(len(self.memory[l3[1]].position)):
                #                 teste_p[256+l3[1]][j] = self.memory[l3[1]].position[j]
                #             teste_p = teste_p.flatten()
                #             cuda.memcpy_htod(self.position_g, teste_p)
                #             # atualizar fitness da memoria nova
                #             zdt1 = self.mod.get_function("zdt1")
                #             zdt1(self.position_g, self.params.position_dim_g, self.fitness_g,
                #                  block=(2 * len(self.population) + len(self.memory), 1, 1), grid=(1, 1, 1))
                #             cuda.Context.synchronize()
                #
                #             # teste memoria
                #             print('teste memoria depois da troca')
                #             teste_p = np.zeros(261 * 10, dtype=np.float64)
                #             cuda.memcpy_dtoh(teste_p, self.position_g)
                #             teste_p.shape = 261, 10
                #             teste_f = np.zeros(261 * 2, dtype=np.float64)
                #             cuda.memcpy_dtoh(teste_f, self.fitness_g)
                #             teste_f.shape = 261, 2
                #             l2 = []
                #             for i in range(len(self.memory)):
                #                 for j in range(len(self.memory[i].position)):
                #                     l2.append(abs(self.memory[i].position[j] - teste_p[i + 256][j]))
                #             l2 = np.array(l2)
                #             l2 = np.where(abs(l2) > 1e-5)[0]
                #             print(l2)
                #             l2 = []
                #             for i in range(len(self.memory)):
                #                 for j in range(len(self.memory[i].fitness)):
                #                     l2.append(abs(self.memory[i].fitness[j] - teste_f[i + 256][j]))
                #             l2 = np.array(l2)
                #             l2 = np.where(abs(l2) > 1e-5)[0]
                #             print(l2)

                if self.generation_count == 7:
                    pass


                self.log_memory = False
                if self.log_memory:
                    file = open(self.log_memory + "fit.txt", "w")
                    # file = open(self.log_memory + "fit.txt", "a+")
                    memory_fitness = ""
                    for m in self.memory:
                        string = ""
                        for i in range(self.params.objectives_dim):
                            string += str(m.fitness[i]) + " "
                        string = string[:-1]
                        memory_fitness += string + ", "
                        # memory_fitness += string + "\n"
                    memory_fitness = memory_fitness[:-2]
                    memory_fitness += "\n"
                    file.write(memory_fitness)
                    file.close()

                    file2 = open(self.log_memory + "pos.txt", "a+")
                    memory_position = ""
                    for m in self.memory:
                        string = ""
                        for i in range(self.params.position_dim):
                            string += str(m.position[i]) + " "
                        string = string[:-1]
                        memory_position += string + ", "
                    memory_position = memory_position[:-2]
                    memory_position += "\n"
                    file2.write(memory_position)
                    file2.close()

                    if self.gpu:
                        file = open(self.log_memory + "fit_gpu.txt", "w")
                        # file = open(self.log_memory + "fit.txt", "a+")
                        memory_fitness = ""
                        inicio = 2*self.params.population_size
                        fim = inicio+self.params.memory_size

                        fitness = np.zeros((2 * self.params.population_size + self.params.memory_size) *
                                                self.params.objectives_dim, dtype=np.float64)
                        cuda.memcpy_dtoh(fitness, self.fitness_g)

                        for j in range(inicio, fim):
                            string = ""
                            for i in range(self.params.objectives_dim):
                                string += str(fitness[j*self.params.objectives_dim+i]) + " "
                            string = string[:-1]
                            memory_fitness += string + ", "
                        memory_fitness = memory_fitness[:-2]
                        memory_fitness += "\n"
                        file.write(memory_fitness)
                        file.close()

                        file2 = open(self.log_memory + "pos_gpu.txt", "a+")
                        memory_position = ""
                        position = np.zeros((2 * self.params.population_size + self.params.memory_size) *
                                           self.params.position_dim, dtype=np.float64)
                        cuda.memcpy_dtoh(position, self.position_g)

                        for j in range(inicio, fim):
                            string = ""
                            for i in range(self.params.position_dim):
                                string += str(position[j*self.params.position_dim+i]) + " "
                            string = string[:-1]
                            memory_position += string + ", "
                        memory_position = memory_position[:-2]
                        memory_position += "\n"
                        file2.write(memory_position)
                        file2.close()

                # if self.gpu:
                #     # file = open(self.log_memory + "fit_gpu.txt", "a")
                #     file = open(self.log_memory + "fit_gpu.txt", "w")
                #     memory_fitness = ""
                #     self.fitness = np.zeros((2 * self.params.population_size + self.params.memory_size) *
                #                             self.params.objectives_dim, dtype=np.float64)
                #     cuda.memcpy_dtoh(self.fitness, self.fitness_g)
                #     self.fitness.shape = 2 * self.params.population_size + self.params.memory_size, self.params.objectives_dim
                #     for m in self.fitness[2 * self.params.population_size:]:
                #         string = ""
                #         for i in range(self.params.objectives_dim):
                #             string += str(m[i]) + " "
                #         string = string[:-1]
                #         memory_fitness += string + ", "
                #     memory_fitness = memory_fitness[:-2]
                #     memory_fitness += "\n"
                #     file.write(memory_fitness)
                #     file.close()
                #
                #     file = open(self.log_memory + "pos_gpu.txt", "w")
                #     # file = open(self.log_memory + "pos_gpu.txt", "a")
                #     memory_position = ""
                #     self.position = np.zeros((2 * self.params.population_size + self.params.memory_size) *
                #                              self.params.position_dim, dtype=np.float64)
                #     cuda.memcpy_dtoh(self.position, self.position_g)
                #     self.position.shape = 2 * self.params.population_size + self.params.memory_size, self.params.position_dim
                #     for m in self.position[2 * self.params.population_size:]:
                #         string = ""
                #         for i in range(self.params.position_dim):
                #             string += str(m[i]) + " "
                #         string = string[:-1]
                #         memory_position += string + ", "
                #     memory_position = memory_position[:-2]
                #     memory_position += "\n"
                #     file.write(memory_position)
                #     file.close()
                #     pass

                # Fim do loop principal.

                delta_evals = self.fitness_eval_count - prev_fitness_eval
                pbar.update(delta_evals)
                prev_fitness_eval = self.fitness_eval_count

                self.generation_count = self.generation_count + 1
                # break
                # print(self.generation_count)
                self.check_stopping_criteria()
                # print('fim iteracao')

                # # teste
                # teste_f = np.zeros(384 * 2, np.float64)
                # cuda.memcpy_dtoh(teste_f, self.fitness_g)
                # teste_f.shape = 384, 2
                # plt.plot(teste_f[:, 0], teste_f[:, 1], 'ro')
                # plt.show()
                pass

                # testes
                # teste_grafico = True
                teste_tempo = False
                if teste_grafico:
                    # print('\nteste fitness-grafico')
                    # if self.generation_count == 7:
                    if self.generation_count > -1:
                        teste4 = np.zeros((2*self.params.population_size+self.params.memory_size)*2, dtype=np.float64)
                        cuda.memcpy_dtoh(teste4, self.fitness_g)
                        teste4.shape = 2*self.params.population_size+self.params.memory_size, 2

                        # teste_f = np.zeros(256, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste_f, self.fronts_g)
                        #
                        # teste_tam = np.zeros(256, dtype=np.int32)
                        # cuda.memcpy_dtoh(teste_tam, self.tams_fronts_g)

                        # fitness_cpu_x = []
                        # fitness_cpu_y = []
                        # for i in self.population:
                        #     fitness_cpu_x.append(i.fitness[0])
                        #     fitness_cpu_y.append(i.fitness[1])

                        # front0_cpu_x = []
                        # front0_cpu_y = []
                        # for i in self.fronts[0]:
                        #     front0_cpu_x.append(i.fitness[0])
                        #     front0_cpu_y.append(i.fitness[1])

                        # l = []
                        # for i in range(len(self.population)):
                        #     l2 = abs(np.array(self.population[i].fitness) - teste4[i])
                        #     l.extend(l2)
                        # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                        # print(l2)
                        # print(max(np.abs(l)))
                        #
                        # l = []
                        # for i in range(len(self.population_copy)):
                        #     l2 = abs(np.array(self.population_copy[i].fitness) - teste4[i + 128])
                        #     l.extend(l2)
                        # l2 = np.where(abs(np.array(l)) > 1e-4)[0]
                        # print(l2)
                        # print(max(np.abs(l)))

                        if func == 'ZDT1':
                            f = get_problem("zdt1")
                        elif func == 'ZDT2':
                            f = get_problem("zdt2")
                        elif func == 'ZDT3':
                            f = get_problem("zdt3")
                        elif func == 'DTLZ1':
                            f = get_problem("dtlz1")
                        elif func == 'MW1':
                            f = get_problem("mw1")
                        pf_a = f.pareto_front(use_cache=False)

                        plt.title(str(self.generation_count))
                        # plt.subplot(1,2,1)
                        # plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro',teste4[:128, 0],
                        #          teste4[:128, 1], 'go',
                        #          teste4[teste_f[0:teste_tam[0]], 0],
                        #          teste4[teste_f[0:teste_tam[0]], 1], 'bo')
                        # plt.axis((0, 1., 0. , 1.5))
                        plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro', teste4[:128, 0],
                                 teste4[:128, 1], 'go',
                                 teste4[256:, 0],
                                 teste4[256:, 1], 'bo')
                        plt.legend(['paretto', 'population', 'memory'])
                        plt.axis((0, 1., 0, 5))

                        # plt.subplot(1, 2, 2)
                        # plt.plot(pf_a[:, 0], pf_a[:, 1], 'ro',
                        #          fitness_cpu_x, fitness_cpu_y, 'go',
                        #          front0_cpu_x, front0_cpu_y, 'bo')
                        # plt.axis((0, 1., 0., 1.5))

                        plt.show()
                        # break

                if teste_tempo:
                    # f = open('teste_tempo', 'r+')
                    f = sys.stdout
                    print('differential mutation', file=f)
                    print('cpu', cpu[0], file=f)
                    print('gpu', gpu[0], file=f)
                    print('gpu/cpu {:0.2f}%\n'.format((gpu[0]/cpu[0])*100), file=f)

                    print('atualizacao fronts testeTempo250624 memory update', file=f)
                    print('cpu', cpu[1], file=f)
                    print('gpu', gpu[1], file=f)
                    print('gpu/cpu {:0.2f}%\n'.format((gpu[1] / cpu[1]) * 100), file=f)

                    print('copia', file=f)
                    print('cpu', cpu[2], file=f)
                    print('gpu', gpu[2], file=f)
                    print('gpu/cpu {:0.2f}%\n'.format((gpu[2] / cpu[2]) * 100), file=f)

                    print('mutate weights', file=f) #possivel gargalo de memoria. Desempenho da gpu caiu muito
                    print('cpu', cpu[3], file=f)
                    print('gpu', gpu[3], file=f)
                    print('gpu/cpu {:0.2f}%\n'.format((gpu[3] / cpu[3]) * 100), file=f)

                    print('global best', file=f)
                    print('cpu', cpu[4], file=f)
                    print('gpu', gpu[4], file=f)
                    print('gpu/cpu {:0.2f}%\n'.format((gpu[4] / cpu[4]) * 100), file=f)

                    print('move+update personal', file=f)
                    print('cpu', cpu[5], file=f)
                    print('gpu', gpu[5], file=f)
                    print('gpu/cpu {:0.2f}%\n'.format((gpu[5] / cpu[5]) * 100), file=f)

                    print('fronts', file=f)
                    print('cpu', cpu[6], file=f)
                    print('gpu', gpu[6], file=f)
                    print('gpu/cpu {:0.2f}%\n'.format((gpu[6] / cpu[6]) * 100), file=f)

                    print('next generation', file=f)
                    print('cpu', cpu[7], file=f)
                    print('gpu', gpu[7], file=f)
                    print('gpu/cpu {:0.2f}%\n'.format((gpu[7] / cpu[7]) * 100), file=f)

                    print('update memory', file=f)
                    print('cpu', cpu[8], file=f)
                    print('gpu', gpu[8], file=f)
                    print('gpu/cpu {:0.2f}%\n'.format((gpu[8] / cpu[8]) * 100), file=f)

                    print('total', file=f)
                    print(teste_tam[0], file=f)
                    print('cpu', sum(cpu), file=f)
                    print('gpu', sum(gpu), file=f)
                    print('gpu/cpu {:0.2f}%'.format((sum(gpu)/ sum(cpu))* 100), file=f)

            # print('total')
            # print(teste_tam[0])
            # print('cpu', sum(cpu))
            # print('gpu', sum(gpu))
            # print('gpu/cpu {:0.2f}%'.format((sum(gpu) / sum(cpu)) * 100))

            f = open('results.pkl', 'rb')
            results = pickle.load(f)
            f.close()

            count = results['count']
            count+=1
            results['count'] = count

            # fitness = np.zeros((2 * self.params.population_size + self.params.memory_size) * 2,
            #                    dtype=np.float64)
            # fitness.shape = 2 * self.params.population_size + self.params.memory_size, 2
            fitness = np.zeros((2 * self.params.population_size + self.params.memory_size) * self.params.objectives_dim,
                               dtype=np.float64)
            cuda.memcpy_dtoh(fitness, self.fitness_g)
            fitness.shape = 2 * self.params.population_size + self.params.memory_size, self.params.objectives_dim

            position = np.zeros((2 * self.params.population_size + self.params.memory_size) * self.params.position_dim,
                               dtype=np.float64)
            cuda.memcpy_dtoh(position, self.position_g)
            position.shape = 2 * self.params.population_size + self.params.memory_size, self.params.position_dim

            cur = np.zeros(1, dtype=np.int32)
            cuda.memcpy_dtoh(cur, self.params.current_memory_size_g)

            fronts = np.zeros(256, dtype=np.int32)
            cuda.memcpy_dtoh(fronts, self.fronts_g)

            tam = np.zeros(256, dtype=np.int32)
            cuda.memcpy_dtoh(tam, self.tams_fronts_g)

            # results[count] = (position, fitness, cur, fronts, tam, teste_dict)
            results[count] = (position, fitness, cur, fronts, tam, self.params.func_n)

            f = open('results.pkl', 'wb')
            pickle.dump(results, f)
            f.close()

            # clear_context_caches()

            return cpu, gpu