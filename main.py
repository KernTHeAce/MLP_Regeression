from Network import MLP
from RK_Method import EquationsSystem
import datasets as ds
from Activations import functions as func
from Visualisation import visual3d, visual3ds, visual3ds_final
import matplotlib.pyplot as plt
import math


def dx(x, y, z):
    return 10 * (y - x)


def dy(x, y, z):
    return 28 * x - y - y * z


def dz(x, y, z):
    return y * x - 8/3 * z


if __name__ == '__main__':
    dataset_size = 2000
    step = 0.01

    input_size = 3

    functions = [func['sigma'], func['line']]

    constants = {
        'alpha': 0.01,
        'lambda': 0.9,
        'beta': 0.99,
        'max_error': 0.000001
    }

    es = EquationsSystem(dx, dy, dz)
    data = es.calculating(2000, 0.01)

    dataset = ds.EquationsSystemDataset(data, 2000, learn=0.6, test=0.4)
    dataset.data_transform()
    dataset.create_sets()

    model = MLP(3, 100, 3, functions)
    model.learning(dataset, constants)
    result = model.forecasting(dataset.full_set, 2000 - 3)

    #visual3ds_final(data, dataset)

    x = []
    y = []
    z = []
    for row in result:
        x.append(row[0])
        y.append(row[1])
        z.append(row[2])

    visual3d(x, y, z)
    visual3ds(data)
    #input()




