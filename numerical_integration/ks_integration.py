# -*- coding: utf-8 -*-

"""
Created on Fri Jun  7 02:16:31 2024
@author: decla
"""
from numerical_integration import ks_etdrk4 as ks
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

num_grid_points = 512
time_step = .25
periodicity_length = 200
IC_bounds = [-.6, .6]
L_22_Lyap_Time = 20.83
IC_seed = 10
params = np.array([[], []], dtype=np.complex128)
font_size = 15.


def int_plot(
        time_range: int,
        periodicity_length: int,
        num_grid_points: int,
        time_step: float = .25,
        params: np.ndarray = np.array([[], []], dtype=np.complex128),
        plot: bool = True,
        IC_seed: int = 11000,
        output: bool = False,
        transient_length: int = 100
        ):
    u_arr, new_params = ks.kursiv_predict(
        d=periodicity_length,
        u0=np.random.default_rng(IC_seed).uniform(IC_bounds[0], IC_bounds[1], size=num_grid_points),
        N=num_grid_points,
        tau=time_step,
        T=time_range + transient_length,
        params=params,
        noise=np.zeros((1, 1), dtype=np.double)
    )

    mean = np.mean(u_arr, axis=1)
    stdev = np.std(u_arr, axis=1)
    u_arr = (u_arr - mean.reshape((-1, 1))) / stdev.reshape((-1, 1))
    u_arr = u_arr[:, transient_length:]
    if plot:
        with mpl.rc_context({"font.size": font_size}):
            fig, ax = plt.subplots(constrained_layout=True)
            ax.set_title("Kursiv_Predict")
            x = np.arange(u_arr.shape[1]) * time_step / L_22_Lyap_Time
            y = np.arange(u_arr.shape[0]) * periodicity_length / num_grid_points
            x, y = np.meshgrid(x, y)
            pcm = ax.pcolormesh(x, y, u_arr)
            ax.set_ylabel("$x$")
            ax.set_xlabel("$t$")
            fig.colorbar(pcm, ax=ax, label="$u(x, t)$")
            plt.show()

    if output:
        np.save(f"X_seed{IC_seed}_L{periodicity_length}_Q{num_grid_points}_T{time_range}_", u_arr)

    return u_arr.copy()


# int_plot(time_range=1000,
#          periodicity_length=periodicity_length,
#          num_grid_points=num_grid_points,
#          IC_seed=IC_seed)