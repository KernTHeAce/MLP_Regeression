import matplotlib.pyplot as plt
import numpy as np


def visual2d(row, y_name = 'data', x_name = 'time'):
    fig1, ax1 = plt.subplots()
    t = []
    i = 0
    while i < ((len(row) + 10) * 0.01):
        t.append(i)
        i += 0.01

    while len(row) != len(t):
        t.pop()

    ax1.plot(t, row)
    ax1.grid()

    ax1.set_xlabel(x_name)
    ax1.set_ylabel(y_name)

    plt.show()


def visual3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='parametric curve')
    plt.show()

    #t = np.arange(0, size * step, step)
