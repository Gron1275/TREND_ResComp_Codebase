# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:19:11 2024

Code from Alex Wikner et al.
https://github.com/awikner/res-noise-stabilization/blob/master/src/res_reg_lmnt_awikner/ks_etdrk4.py
"""

import numpy as np
from numpy.fft import fft, ifft


from numba import jit, prange, objmode

@jit(nopython = True, fastmath = True, parallel = True)
def mean_numba_axis1(mat):

    res = np.zeros(mat.shape[0])
    for i in prange(mat.shape[0]):
        res[i] = np.mean(mat[i])

    return res

@jit(nopython = True, fastmath = True)
def precompute_KS_params(N, d, tau, M = 16, const = 0):
    k = np.concatenate((np.arange(int(N/2)), np.arange(-int(N/2), 0)))*2*np.pi/d
    L = (1+const)*k**2.0 - k**4.0
    E = np.exp(tau*L)
    E2 = np.exp(tau/2*L)
    r = np.exp(1j * np.pi * (np.arange(1, M+1)-0.5)/M)
    LR = tau*(np.zeros((1,M)) + L.reshape(-1,1)) + (np.zeros((N,1)) + r)
    Q  = tau*mean_numba_axis1(np.real((np.exp(LR/2)-1)/LR))
    f1 = tau*mean_numba_axis1(np.real((-4-LR+np.exp(LR)*(4-3*LR+LR**2.0))/(LR**3.0)))
    f2 = tau*mean_numba_axis1(np.real((2+LR+np.exp(LR)*(-2+LR))/(LR**3.0)))
    f3 = tau*mean_numba_axis1(np.real((-4-3*LR-LR**2.0+np.exp(LR)*(4-LR))/(LR**3.0)))
    g  = -0.5*1j*k
    params = np.zeros((7,N), dtype = np.complex128)
    params[0] = E
    params[1] = E2
    params[2] = Q
    params[3] = f1
    params[4] = f2
    params[5] = f3
    params[6] = g

    return params


@jit(nopython = True, fastmath = True)
def kursiv_forecast(u, params, noise = np.zeros(1, dtype = np.double)):

    with objmode(unext = 'double[:]'):
        v  = fft(u + noise)
        Nv = params[6]*fft(np.real(ifft(v))**2.0)
        a  = params[1]*v + params[2]*Nv
        Na = params[6]*fft(np.real(ifft(a))**2.0)
        b  = params[1]*v + params[2]*Na
        Nb = params[6]*fft(np.real(ifft(b))**2.0)
        c  = params[1]*a + params[2]*(2*Nb - Nv)
        Nc = params[6]*fft(np.real(ifft(c))**2.0)
        vnext  = params[0]*v + Nv*params[3] + 2*(Na+Nb)*params[4] + Nc*params[5]
        unext = np.real(ifft(vnext))
    return unext

@jit(nopython = True, fastmath = True)
def kursiv_forecast_pred(u, params, noise = np.zeros((1,1), dtype = np.double)):
    u = np.ascontiguousarray(u.T)
    with objmode(unext = 'double[:,:]'):
        v  = fft(u + noise,axis = 1)
        Nv = params[6]*fft(np.real(ifft(v, axis = 1))**2.0, axis = 1)
        a  = params[1]*v + params[2]*Nv
        Na = params[6]*fft(np.real(ifft(a, axis = 1))**2.0, axis = 1)
        b  = params[1]*v + params[2]*Na
        Nb = params[6]*fft(np.real(ifft(b, axis = 1))**2.0, axis = 1)
        c  = params[1]*a + params[2]*(2*Nb - Nv)
        Nc = params[6]*fft(np.real(ifft(c, axis = 1))**2.0, axis = 1)
        v  = params[0]*v + Nv*params[3] + 2*(Na+Nb)*params[4] + Nc*params[5]
        unext = np.real(ifft(v, axis = 1))
    return unext.T


@jit(nopython = True, fastmath = True)
def kursiv_predict(u0, tau = 0.25, N = 64, d = 22, T = 100, params = np.array([[],[]], dtype = np.complex128),
                   int_steps = 1, noise = np.zeros((1,1), dtype = np.double)):
    if params.size == 0:
        new_params = precompute_KS_params(N, d, tau)
    else:
        new_params = params
    steps = T*int_steps


    u_arr = np.zeros((N, steps+int_steps))
    u_arr[:,0] = u0
    if noise.size == 1 and noise[0,0] == 0.:
        for i in range(steps+int_steps-1):
            u_arr[:,i+1] = kursiv_forecast(u_arr[:,i], new_params)
    else:
        for i in range(steps+int_steps-1):
            u_arr[:,i+1] = kursiv_forecast(u_arr[:,i], new_params, noise[:,i])
    return np.ascontiguousarray(u_arr[:,::int_steps]), new_params

@jit(nopython = True, fastmath = True)
def kursiv_predict_pred(u0_array, tau = 0.25, N = 64, d = 22, T = 100, params = np.array([[],[]], dtype = np.complex128),
                        noise = np.zeros((1,1), dtype = np.double)):
    if params.size == 0:
        new_params = precompute_KS_params(N, d, tau)
    else:
        new_params = params
    if noise.size == 1 and noise[0, 0] == 0.:
        u_arr = kursiv_forecast_pred(u0_array, new_params)
    else:
        u_arr = kursiv_forecast_pred(u0_array, new_params, noise)
    return u_arr, new_params