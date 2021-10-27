import numpy as np
import math


def sigma(S):
    return float(1 / (1 + np.exp(-S)))


def d_sigma(Y):
    return sigma(Y) * (1 - sigma(Y))


def line(S):
    return S


def d_line(Y):
    return 1


def ReLU(S):
    if S > 0:
        return S
    else:
        return -3 * S


def d_ReLU(Y):
    if Y > 0:
        return 1
    else:
        return -3


def bin_sigma(S):
    return (2 / (1 + math.exp(-S))) - 1


def d_bin_sigma(Y):
    #return 0.5 * (1 - Y ** 2)
    return (math.exp(Y) - 1) / (math.exp(Y) - 1)


functions = {
    'sigma': [sigma, d_sigma],
    'line': [line, d_line],
    'relu': [ReLU, d_ReLU],
    'bin_sigma': [bin_sigma, d_bin_sigma]}