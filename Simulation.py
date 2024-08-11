# -*- coding: utf-8 -*-

"""
Created on Sun Aug  11 11:23:31 2024
@author: grennon
"""

import numpy as np
import time
import generate_reservoir
from reservoir_computing_schemes import rc_multi_output, rc_single_output
from prediction_files import predict_multi_output_layer, predict_single_output_layer, prediction_analysis

"""
BEGIN GLOBAL PARAMETERS
"""
# X_SEED = 1000
square_nodes = False
time_step = 0.25
prediction_steps = 1000
transient_length = 500
batch_length = 1000
TRAINLENGTH = 35000
num_grid_points = 256
periodicity_length = 100
reservoir_size = 2000
num_reservoirs = 16
p = [0.6, 3, 0.1]  # spectral radius, degree, input scaling
"""
END GLOBAL PARAMETERS
"""


class Simulation:

    def __init__(self, X: np.ndarray, beta: float, train_length: int = TRAINLENGTH, noise_amplitude = 0, overlap = -1):
        self.X = X
        self.square_nodes = False

        self.res_params = {
            'N': reservoir_size,
            'num_inputs': num_grid_points,
            'radius': 0.6,
            'degree': 3,
            'nonlinear_func': np.tanh,
            'sigma': 0.1,
            'bias': 1.3,
            'overlap': 6,
            'inputs_per_reservoir': 16
        }
        if overlap != -1:
            self.res_params['overlap'] = overlap
        self.res_params['train_length'] = train_length
        self.beta = beta
        self.res_seed = 0
        self.noise_amplitude = noise_amplitude
        self.reservoir = generate_reservoir.erdos_renyi_reservoir(size=self.res_params['N'], degree=self.res_params['degree'], radius=self.res_params['radius'], seed=self.res_seed)
        self.in_weight = generate_reservoir.generate_W_in(num_inputs=self.res_params['inputs_per_reservoir'] + 2 * self.res_params['overlap'], res_size=self.res_params['N'], sigma=self.res_params['sigma'],seed=self.res_seed)

        self.training_time = 0
        self.prediction_time = 0

    def single_train(self):
        global transient_length, batch_length
        discard_len = transient_length
        batch_len = batch_length
        start_train = time.time()

        self.W_out, self.reservoirs_states = rc_single_output.fit_output_weight(resparams=self.res_params, input=self.X, reservoir=self.reservoir, W_in=self.in_weight, discard=discard_len, batch_len=batch_len, square_nodes=self.square_nodes, beta=self.beta)

        self.training_time = time.time() - start_train
        print(f"Training done for {self.beta}")

    def multi_train(self):
        global transient_length, batch_length
        discard_len = transient_length
        batch_len = batch_length
        start_train = time.time()
        self.out_weights = list()
        self.reservoirs_states = list()
        for i in range(self.res_params['num_inputs'] // self.res_params['inputs_per_reservoir']):

            loc = chop_data(self.X[:, :self.res_params['train_length']], self.res_params['inputs_per_reservoir'], self.res_params['overlap'], i)
            print(f'Loc shape: {loc.shape}')

            W_out, res_state = rc_multi_output.fit_output_weights(
                resparams=self.res_params,
                W_in=self.in_weight,
                reservoir=self.reservoir,
                data=loc[self.res_params['overlap']: -self.res_params['overlap'], :self.res_params['train_length']],
                batch_len=batch_len,
                loc=loc,
                discard=discard_len,
                square_nodes=square_nodes,
                beta=self.beta
            )
            print(f'Training stage {(i + 1) / (self.res_params["num_inputs"] // self.res_params["inputs_per_reservoir"]) * 100} % done')
            self.reservoirs_states.append(res_state)
            self.out_weights.append(W_out)

        self.training_time = time.time() - start_train

    def predict_single(self, prediction_steps):
        start_predict = time.time()

        t_pred = np.linspace(0, prediction_steps * time_step - time_step, prediction_steps)
        t_pred /= 11.11  # Lyapunov time for L=200
        self.predictions = predict_single_output_layer.parallel_predict(out_weight=self.W_out, reservoir=self.reservoir, in_weight=self.in_weight, final_res_states=self.reservoirs_states, time_steps=prediction_steps, resparams=self.res_params, square_nodes=self.square_nodes)
        self.__vpt = prediction_analysis.valid_time(self.predictions, self.X[:, self.res_params['train_length']: self.res_params['train_length'] + prediction_steps], t_pred)

        self.prediction_time = time.time() - start_predict

    def predict_multi(self, prediction_steps):
        start_predict = time.time()
        t_pred = np.linspace(0, prediction_steps * time_step - time_step, prediction_steps)
        t_pred /= 11.11  # Lyapunov time for L=200
        self.predictions = predict_multi_output_layer.parallel_predict(out_weights=self.out_weights, reservoir=self.reservoir,
                                                                       in_weight=self.in_weight,
                                                                       final_res_states=self.reservoirs_states,
                                                                       time_steps=prediction_steps,
                                                                       resparams=self.res_params,
                                                                       square_nodes=self.square_nodes)
        self.__vpt = prediction_analysis.valid_time(self.predictions, self.X[:, self.res_params['train_length']: self.res_params['train_length'] + prediction_steps],
                                                    t_pred)

        self.prediction_time = time.time() - start_predict

    def synchronize(self, inputs: np.ndarray):
        chunked_inputs = list()

        for i in range(self.res_params['num_inputs']//self.res_params['inputs_per_reservoir']):
            chunked_inputs.append(chop_data(inputs, self.res_params['inputs_per_reservoir'],self.res_params['overlap'], i))
        print(f'Chunk size: {len(chunked_inputs)}')
        tmp = list()
        for i, chunk in enumerate(chunked_inputs):
            self.reservoirs_states[i] = rc_multi_output.reservoir_layer(
                self.reservoir,
                self.in_weight,
                chunk,
                self.res_params,
                batch_len=0,  # Doesn't matter
                discard_length=0
            )[-1]
            tmp.append(rc_multi_output.reservoir_layer(
                self.reservoir,
                self.in_weight,
                chunk,
                self.res_params,
                batch_len=0,  # Doesn't matter
                discard_length=0
            )[-1])
        return tmp

    @property
    def vpt(self):
        return self.__vpt

    @vpt.setter
    def vpt(self, VPT):
        if VPT is float:
            self.__vpt = VPT
        else:
            pass

    @property
    def runtime(self):
        return self.training_time + self.prediction_time

    def save_to_file(self, filename):
        """TODO"""
        return None


def chop_data(data, IPR, overlap, step):
    index = np.arange(data.shape[0])
    return data[np.roll(index, -IPR * step + overlap)[0: IPR + 2 * overlap], :].copy()
