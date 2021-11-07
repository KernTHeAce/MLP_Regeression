import random
import numpy as np
import PDFNetPython3 as pdf3
from Network import MLP
from RK_Method import EquationsSystem
import datasets as ds
from Activations import functions as func
from Visualisation import visual3d, visual2d, visual2d1
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages


def report_to_file(path, number, beta, lambd, response, validation_error, test_error):
    file = open(path, 'a')
    line1 = 'number: {} \nbeta: {}   lambda: {} \n cause: {}\n'.format(number, beta, lambd, response)
    line2 = 'validation: {}    test: {}\n'.format(validation_error, test_error)
    file.write(line1)
    file.write(line2)
    file.write('**********\n\n')
    file.close()


def dx(x, y, z):
    return 10 * (y - x)


def dy(x, y, z):
    return 28 * x - y - y * z


def dz(x, y, z):
    return y * x - 8/3 * z


if __name__ == '__main__':

    print()
    dataset_size = 1000
    step = 0.01

    input_size = 3

    functions = [func['sigma'], func['line']]

    es = EquationsSystem(dx, dy, dz)
    data = es.calculating(dataset_size, 0.01)

    dataset = ds.EquationsSystemDataset(data, dataset_size, input_size=input_size)
    dataset.data_transform()
    dataset.create_sets()

    # constants = {
    #     'alpha': 0.01,
    #     'lambda': 0.0,
    #     'beta': 0.3,
    #     'max_error': 0.001,
    #     'validation_error': 0.0000001
    # }
    # model = MLP(input_size, 15, 3, functions)
    # response = model.learning(dataset, constants)
    # result = model.fit(dataset.full_set, dataset_size - input_size)
    #
    # x = [row[0] for row in result]
    # y = [row[1] for row in result]
    # z = [row[2] for row in result]
    # while len(dataset.data['x']) != len(x):
    #     dataset.data['x'].pop()
    #     dataset.data['y'].pop()
    #     dataset.data['z'].pop()
    #
    # t = np.linspace(0, len(dataset.data['x']), len(dataset.data['x']))
    #
    # visual3d(x, y, z)
    #
    # figure = plt.figure(figsize=(12, 12))
    # axes = figure.subplots(3, 2)
    # axes[0, 0].plot(t, dataset.data['x'], '-', t, x, ':')
    # axes[0, 0].grid()
    # axes[1, 0].plot(t, dataset.data['y'], '-', t, y, ':')
    # axes[1, 0].grid()
    # axes[2, 0].plot(t, dataset.data['z'], '-', t, z, ':')
    # axes[2, 0].grid()
    #
    # val_list = model.report.square_validation_error_list
    # t = np.linspace(0, len(val_list), len(val_list))
    # axes[0, 1].plot(t, val_list)
    # axes[0, 1].grid()
    #
    # learn_list = model.report.square_learning_error_list
    # t = np.linspace(0, len(learn_list), len(learn_list))
    # axes[1, 1].plot(t, learn_list)
    # axes[1, 1].grid()
    #
    # test_list = model.report.square_test_error_list
    # t = np.linspace(0, len(test_list), len(test_list))
    # axes[2, 1].plot(t, test_list)
    # axes[2, 1].grid()
    #
    # plt.show()

    # beta1 = np.linspace(0, 0.999, num=10)
    # lambda1 = np.linspace(0, 0.999, num=10)
    # path = 'reports\\vtl100.pdf'
    # pdf = PdfPages(path)
    # i = 0
    #
    # for beta in beta1:
    #     for lambd in lambda1:
    #         constants = {
    #             'alpha': 0.01,
    #             'lambda': lambd,
    #             'beta': beta,
    #             'max_error': 0.001,
    #             'validation_error': 0.000001
    #         }
    #         model = MLP(input_size, 15, 3, functions)
    #         response = model.learning(dataset, constants)
    #         result, E = model.fit(dataset.full_set, dataset_size - input_size)
    #
    #         print(str(i))
    #         i += 1
    #
    #         x = [row[0] for row in result]
    #         y = [row[1] for row in result]
    #         z = [row[2] for row in result]
    #
    #         while len(dataset.data['x']) != len(x):
    #             dataset.data['x'].pop()
    #             dataset.data['y'].pop()
    #             dataset.data['z'].pop()
    #
    #         # Создаем фигуру с несколькими осями.
    #         figure = plt.figure(figsize=(12, 12))
    #         axes = figure.subplots(3, 2)
    #
    #         line = 'beta: {}   lambda: {}'.format(beta, lambd)
    #         suptitle = figure.suptitle(line)
    #
    #         t = np.linspace(0, len(dataset.data['x']), len(dataset.data['x']))
    #
    #         axes[0, 0].plot(t, dataset.data['x'], '-', t, x, ':')
    #         axes[0, 0].grid()
    #         axes[1, 0].plot(t, dataset.data['y'], '-', t, y, ':')
    #         axes[1, 0].grid()
    #         axes[2, 0].plot(t, dataset.data['z'], '-', t, z, ':')
    #         axes[2, 0].grid()
    #
    #         val_list = model.report.square_validation_error_list
    #         t = np.linspace(0, len(val_list), len(val_list))
    #         axes[0, 1].plot(t, val_list)
    #         axes[0, 1].grid()
    #
    #         learn_list = model.report.square_learning_error_list
    #         t = np.linspace(0, len(learn_list), len(learn_list))
    #         axes[1, 1].plot(t, learn_list)
    #         axes[1, 1].grid()
    #
    #         test_list = model.report.square_test_error_list
    #         t = np.linspace(0, len(test_list), len(test_list))
    #         axes[2, 1].plot(t, test_list)
    #         axes[2, 1].grid()
    #
    #         report_to_file('reports\\vtl_report.txt', i, beta, lambd, response, model.report.get_validation_average_error(), E)
    #         # Сохраняем страницу
    #         pdf.savefig(figure)
    #
    # print(path)
    # pdf.close()
