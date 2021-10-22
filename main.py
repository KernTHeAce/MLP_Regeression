from Network import MLP
from RK_Method import EquationsSystem
from Training_Report import MLPDataSet as ds
from Activations import functions as func
from Visualisation import visual2d, visual3d


def dx(x, y, z):
    return 10 * (y - x)


def dy(x, y, z):
    return 28 * x - y - y * z


def dz(x, y, z):
    return y * x - 8/3 * z


if __name__ == '__main__':
    dataset_size = 2000
    step = 0.01

    functions = [func['sigma'], func['line']]

    constants = {
        'alpha': 0.01,
        'lambda': 0.1,
        'beta': 0.0,
        'max_error': 0.001
    }

    es = EquationsSystem(dx, dy, dz)

    network = MLP(3, 50, 3, functions)

    data_dict = es.calculating(dataset_size, step)

    dataset = ds(data_dict)
    network.learning(dataset, constants)
    network.report.error_visual()
    result = network.forecasting(dataset.test_set)

    x = []
    y = []
    z = []
    for i in range(len(dataset.test_set)):
        x.append(dataset.test_set[i][0])
        y.append(dataset.test_set[i][1])
        z.append(dataset.test_set[i][2])


    visual2d(x)
    visual2d(y)
    visual2d(z)

    visual3d(x, y, z)

    for i in range(len(result)):
        x[i] = (result[i][0])
        y[i] = (result[i][1])
        z[i] = (result[i][2])

    visual2d(x)
    visual2d(y)
    visual2d(z)

    visual3d(x, y, z)

    visual2d(data_dict['x'])
    visual2d(data_dict['y'])
    visual2d(data_dict['z'])

    visual3d(data_dict['x'], data_dict['y'], data_dict['z'])

