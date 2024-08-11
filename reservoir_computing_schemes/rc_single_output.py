import numpy as np


def chop_data(data, IPR, overlap, step):
    index = np.arange(data.shape[0])
    return data[np.roll(index, -IPR * step + overlap)[0: IPR + 2 * overlap], :].copy()


def reservoir_layer(A, W_in, input, resparams, batch_len, discard_length, in_res_states=None):
    N = resparams['N']
    g = resparams['nonlinear_func']
    bias = resparams['bias']
    if in_res_states is not None:
        res_states = in_res_states[:, None].repeat(input.shape[1],axis=1)
    else:
        res_states = np.zeros((N, input.shape[1]))

    for i in range(input.shape[1] - 1):
        res_states[:, i + 1] = g(A @ res_states[:, i] + W_in @ input[:, i] + bias)
    return res_states[:, discard_length:].copy()


def fit_output_weight(resparams, input, reservoir, W_in, discard, batch_len=1000, square_nodes: bool = True, beta: float = 0.001):
    N = resparams['N']
    train_length = resparams['train_length']
    IPR = resparams['inputs_per_reservoir']
    num_inputs = resparams['num_inputs']
    overlap = resparams['overlap']

    SS_T = np.zeros((N, N))
    VS_T = np.zeros((IPR, N))

    reservoir_states = list()
    for chunk in range(num_inputs // IPR):

        loc = chop_data(input, IPR, overlap, chunk)
        data = loc[overlap: -overlap, :]
        # is repeatedly cutting at training_length necessary in the two lines below? isn't that done in data above?
        V_batches = np.split(data[:, batch_len + discard+1: train_length], np.arange(0, data[:, batch_len + discard+1: train_length].shape[1], batch_len), axis=1)
        S_batches = np.split(loc[:, batch_len + discard: train_length-1], np.arange(0, loc[:, batch_len + discard: train_length-1].shape[1], batch_len), axis=1)
        # V_batches = np.split(data[:, batch_len + discard: train_length],
        #                      np.arange(0, data[:, batch_len + discard: train_length].shape[1], batch_len), axis=1)
        # S_batches = np.split(loc[:, batch_len + discard: train_length],
        #                      np.arange(0, loc[:, batch_len + discard: train_length].shape[1], batch_len), axis=1)
        # print(f'V_batch arange: {np.arange(0, data[:, batch_len + discard + 1: train_length].shape[1], batch_len)}')
        # print(f'S_batch arange: {np.arange(0, loc[:, batch_len + discard: train_length - 1].shape[1],batch_len)}')
        # INDEXING CHANGES MADE 7/30/24 2:37
        # first_S = loc[:, :batch_len + discard -1]

        S_batches[0] = loc[:, :batch_len + discard]
        V_batches[0] = data[:, discard+1: batch_len + discard+1]
        # S_batches[0] = loc[:, :batch_len + discard]
        # V_batches[0] = data[:, discard: batch_len + discard]
        res_state = reservoir_layer(reservoir, W_in, S_batches[0], resparams, batch_len=batch_len, discard_length=discard)

        if square_nodes:
            res_state[::2, :] = np.square(res_state[::2, :].copy())

        SS_T = SS_T + res_state @ res_state.T
        VS_T = VS_T + V_batches[0] @ res_state.T

        for i in range(1, len(S_batches)):
            res_state = reservoir_layer(reservoir, W_in, S_batches[i], resparams, batch_len=batch_len, discard_length=0, in_res_states=res_state[:,-1])
            if square_nodes:
                res_state[::2, :] = np.square(res_state[::2, :].copy())
            SS_T += res_state @ res_state.T
            VS_T += V_batches[i] @ res_state.T

        reservoir_states.append(res_state[:, -1].copy())

        print(f'Training stage {(chunk + 1) / (num_inputs // IPR) * 100} % done')

    W_out = VS_T @ np.linalg.pinv(SS_T + beta * (train_length - 1) * num_inputs//IPR * np.identity(N))
    return W_out, reservoir_states
