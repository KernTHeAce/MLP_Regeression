import numpy as np


def sigma_activation(S):
    return float(1 / (1 + np.exp(-S)))


def d_sigma_activation(Y):
    return Y * (1 - Y)


def line_activation(S):
    return S


def d_line_activation(Y):
    return 1


def ReLU(S):
    if S > 0:
        return S
    else:
        return 0


def d_ReLU(Y):
    if Y > 0:
        return 1
    else:
        return 0


functions = {
    'sigma': [sigma_activation, d_sigma_activation],
    'line': [line_activation, d_line_activation],
    'relu': [ReLU, d_ReLU]}