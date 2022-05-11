from Network import MLP
from RK_Method import EquationsSystem
import datasets as ds
from Activations import functions as func
import Visualisation as Visual



def dx(x, y, z):
    return y


def dy(x, y, z):
    return z


def dz(x, y, z):
    return -2.5 * z + y ** 2 - x


if __name__ == '__main__':
    dataset_size = 2000
    step = 0.01
    input_size = 3
    functions = [func['sigma'], func['line']]

    es = EquationsSystem(dx, dy, dz)
    data = es.calculating(dataset_size, step)

    dataset = ds.EquationsSystemDataset(data, dataset_size, input_size=input_size, learn=0.25, validation=0.05)
    dataset.data_transform()
    dataset.create_sets()
    # Visual.visual_data(dataset)
    constants = {
        'alpha': 0.01,
        'hidden alpha': 0.01,
        'lambda': 0.0,
        'beta': 0.333,
        'hidden lambda': 0.0,
        'hidden beta': 0.333,
        'max_error': 0.1,
        'max_first_step_error': 27,
        'max_second_step_error': 10,
        'validation_error': 0.01
    }



    model = MLP(input_size, 18, 3, functions)
    # model.step_by_step_training(dataset, constants)

    response = model.learning(dataset, constants)
    result1, E1 = model.fit(dataset.learning_set, dataset.learn_size)
    result, E = model.fit(dataset.full_set, dataset_size - input_size)

    print(response)
    print('learning error: {0}     full error: {1}'.format(E1, E))
    Visual.visual_res(dataset, result, 'forecasting')
    Visual.visual_res(dataset, result1, 'approximation')
