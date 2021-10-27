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


def visual3ds(data):
    x = data['x'][:]
    y = data['y'][:]
    z = data['z'][:]

    visual3d(x, y, z)


def visual3ds_final(data1, data2):
    x1 = data1['x'][:]
    y1 = data1['y'][:]
    z1 = data1['z'][:]

    x2 = []
    y2 = []
    z2 = []
    for row in data2.full_set['x']:
        x2.append(row[0])
        y2.append(row[1])
        z2.append(row[2])

    visual3d_final(x1, y1, z1, x2, y2, z2)


def visual3d_final(x, y, z, x1, y1, z1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, '--', x1, y1, z1, ':', label='parametric curve')
    plt.show()



