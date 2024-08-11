#from memory_profiler import profile
import numpy as np


def reservoir_layer(A, W_in, input, resparams, batch_len, discard_length, in_res_states=None):
    N = resparams['N']
    g = resparams['nonlinear_func']
    bias = resparams['bias']

    if in_res_states is not None:
        res_states = in_res_states[:, None].repeat(input.shape[1], axis=1)
    else:
        res_states = np.zeros((N, input.shape[1]))

    for i in range(input.shape[1] - 1):
        res_states[:, i + 1] = g(A @ res_states[:, i] + W_in @ input[:, i] + bias)
    return res_states[:, discard_length:].copy()


def fit_output_weights(resparams, data, reservoir, W_in, loc, discard, batch_len=1000, batch=True, square_nodes: bool = True, beta: float = 0.0001):

    N = resparams['N']
    train_length = resparams['train_length']
    if batch:

        V_batches = np.split(
            data[:, batch_len + discard: train_length],
            np.arange(0, data[:, batch_len + discard: train_length].shape[1], batch_len),
            axis = 1
            )
        temp_fix = np.split(
            loc[:, batch_len + discard: train_length],
            np.arange(0, loc[:, batch_len + discard: train_length].shape[1], batch_len),
            axis = 1
            )
        first_temp = loc[:, :batch_len + discard]

        temp_fix[0] = first_temp

        V_batches[0] = data[:, discard: batch_len + discard]

        res_state = reservoir_layer(
            reservoir,
            W_in,
            first_temp,
            resparams,
            batch_len = batch_len,
            discard_length = discard
            )  # pass inputs here
        if square_nodes:
            res_state[::2,:] = np.square(res_state[::2,:].copy())
        SS_T = res_state @ res_state.T
        #print(f'v batch size: {V_batches[0].shape}')
        #print(f'res state size: {res_state.T.shape}')
        VS_T = V_batches[0] @ res_state.T
        #print(f'Initial  size idl: {data[:,batch_len:(batch_len+discard)].shape}')
        ##VS_T = data[:, batch_len: batch_len + discard)] @ res_state.T
        # print(f'Number of batches: {train_length//batch_len}')
        # print(f'V batches size: {len(V_batches)}')
        # print(f'temp fix size: {len(temp_fix)}')
        for i in range(1, len(temp_fix)): #(train_length // batch_len) - 2):
            # print(V_batches[i+1].shape)
            # res_state = reservoir_layer(reservoir, W_in, V_batches[i], res_params, batch_len)
            res_state = reservoir_layer(reservoir, W_in, temp_fix[i], resparams, batch_len, 0,in_res_states=res_state[:,-1])
            if square_nodes:
                res_state[::2, :] = np.square(res_state[::2, :].copy())
            SS_T += res_state @ res_state.T
            VS_T += V_batches[i] @ res_state.T

    else:
        res_states = reservoir_layer(reservoir, W_in, data, resparams, batch_len=resparams['train_length'])
        SS_T = res_states @ res_states.T
        VS_T = data @ res_states.T
        res_state = res_states


    # Tikhonov ridge regression using normal equations
    # W_out = VS_T @ np.linalg.pinv(SS_T + beta * np.identity(N))

    """
    Output weight layer with scaling factor below
    """

    W_out = VS_T @ np.linalg.pinv(SS_T + beta * (train_length - 1) * np.identity(N))

    # W_out = data @ res_states.T @ np.linalg.pinv(res_states @ res_states.T + beta * np.identity(N))
    return W_out, res_state[:, -1].copy() # returns w_out, and then an NX1 vector I believe


def predict(W_out, A, W_in, training_res_state, time_steps, resparams):
    #predict_length parameter
    """Closed loop prediction"""
    g = resparams['nonlinear_func']
    bias = resparams['bias']
    final_res_state = training_res_state
    # final_res_state = training_res_states[:, 1000]

    # predictions = np.zeros((W_out.shape[0], res_params['train_length']-1000))
    predictions = np.zeros((W_out.shape[0], time_steps))
    rt = final_res_state
    # for i in range(res_params['train_length']-1000):
    #     predictions[:, i] = W_out @ rt
    #     rt = g(A @ rt + W_in @ predictions[:, i] + bias)
    for i in range(time_steps):
        predictions[:, i] = W_out @ rt
        rt = g(A @ rt + W_in @ predictions[:, i] + bias)

    return predictions

def open_train_fit(W_out, A, W_in, training_res_states, time_steps, resparams,X):
    """Open loop prediction"""
    g = resparams['nonlinear_func']
    bias = resparams['bias']
    beginning_res_state = training_res_states[:,0]
    print(f'Beginning res shape{beginning_res_state.shape}')
    print(f'A shape {A.shape}')
    predictions = np.zeros((W_out.shape[0], resparams['train_length']))
    attempt = np.zeros((W_out.shape[0], resparams['train_length']))
    print(f'Attempt shape {attempt.shape}')
    rt = beginning_res_state
    for i in range(resparams['train_length']):
        predictions[:,i] = W_out @ rt
        attempt[:,i] = W_out @ g(A @ rt + W_in @ predictions[:,i] + bias)
        rt = training_res_states[:,i + 1]
    return predictions, attempt

def open_predict(W_out, A, W_in, input_signal, initial_state, time_steps, resparams,X):
    """Open loop prediction"""
    g = resparams['nonlinear_func']
    bias = resparams['bias']
    beginning_res_state = initial_state
    print(f'Beginning res shape{beginning_res_state.shape}')
    print(f'A shape {A.shape}')
    predictions = np.zeros((W_out.shape[0], resparams['train_length']))
    attempt = np.zeros((W_out.shape[0], resparams['train_length']))
    print(f'Attempt shape {attempt.shape}')
    rt = beginning_res_state
    for i in range(resparams['train_length']-1):
        predictions[:,i] = W_out @ rt
        rt = g(A @ rt + W_in @ input_signal[:, i] + bias)
        attempt[:,i] = W_out @ g(A @ rt + W_in @ predictions[:,i] + bias)
    return predictions, attempt
