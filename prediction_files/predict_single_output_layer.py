import numpy as np


def chop_data(data, n, m, step):
    index = np.arange(data.shape[0])
    # do this so it doesnt hafta query data. prolly should just have the parameter as num_inputs
    return data[np.roll(index, -n*step + m)[0:n+2*m], :]


def parallel_predict(out_weight, reservoir, in_weight, final_res_states, time_steps, resparams, alpha=1, square_nodes: bool = True):
    g = resparams['nonlinear_func']
    bias = resparams['bias']
    num_reservoirs = resparams['num_inputs'] // resparams['inputs_per_reservoir']
    print(time_steps)
    # create a (num_reservoirs X Out weights X time step) matrix
    predictions_par = np.zeros((num_reservoirs, out_weight.shape[0], time_steps))
    print(f'Num reservoirs: {num_reservoirs}')
    # for res_state in training_res_states:
    #     # final_res_states.append(res_state[:, -1])
    #     final_res_states.append(res_state)
    # instead of bringing in the whole of training states, the current fit_out_weight func & reservoir_layer
    # is just grabbing and returning the very last entry for each bunch
    for dt in range(time_steps):
        print(f'dt: {dt}')
        for i in range(num_reservoirs):

            predictions_par[i, :, dt] = out_weight @ final_res_states[i]
        whole_preds = predictions_par.copy()
        whole_preds = whole_preds.reshape((resparams['num_inputs'],time_steps))
        for i in range(num_reservoirs):
            # final_res_states[i] = (1 - alpha) * final_res_states[i] + alpha * g(reservoir @ final_res_states[i] + in_weight @ predictions_par[i, :, dt] + bias)
            final_res_states[i] = (1 - alpha) * final_res_states[i] + alpha * g(reservoir @ final_res_states[i] + in_weight @ chop_data(whole_preds, resparams['inputs_per_reservoir'], resparams['overlap'], i)[:,dt] + bias)
            if square_nodes:
                final_res_states[i][::2] = np.square(final_res_states[i][::2].copy())

    return whole_preds
