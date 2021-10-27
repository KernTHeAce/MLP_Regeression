from Network import MLP
from RK_Method import EquationsSystem
import datasets as ds
from Activations import functions as func
from Visualisation import visual2d, visual3ds
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
        'lambda': 0.5,
        'beta': 0.99,
        'max_error': 0.001
    }

    data = EquationsSystem(dx, dy, dz)
    dataset = ds.EquationsSystemDataset(data, 2000)

    visual3ds(dataset)
    input()
    #
    # network = MLP(input_size, 8, 1, functions)
    #
    # dataset = ds(func1, dataset_size, 3)
    # dataset.create_dataset(0.1)
    #
    # visual2d(dataset.learning_set['e'])



