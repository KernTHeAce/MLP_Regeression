import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def visual2d(row, y_name='data', x_name='time'):
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


def visual_res(dataset, result, title='result'):
    x_data = dataset.data['x']
    y_data = dataset.data['y']
    z_data = dataset.data['z']
    x = [row[0] for row in result]
    y = [row[1] for row in result]
    z = [row[2] for row in result]
    while len(x_data) != len(x):
        x_data.pop()
        y_data.pop()
        z_data.pop()

    t = np.linspace(0, len(x_data), len(x_data))

    visual3d(x, y, z)

    figure = plt.figure(figsize=(12, 12))
    axes = figure.subplots(3, 2)
    suptitle = figure.suptitle(title)

    axes[0, 0].plot(t, x_data, '-', t, x, ':')
    axes[0, 0].grid()
    axes[1, 0].plot(t, y_data, '-', t, y, ':')
    axes[1, 0].grid()
    axes[2, 0].plot(t, z_data, '-', t, z, ':')
    axes[2, 0].grid()

    axes[0, 1].plot(x_data, y_data, '-', x, y, ':')
    axes[0, 1].grid()
    axes[1, 1].plot(x_data, z_data, '-', x, z, ':')
    axes[1, 1].grid()
    axes[2, 1].plot(y_data, z_data, '-', y, z, ':')
    axes[2, 1].grid()

    plt.show()


def visual_data(dataset, title='dataset'):
    x_data = dataset.data['x']
    y_data = dataset.data['y']
    z_data = dataset.data['z']

    # x_data = copy.deepcopy(dataset.data['x'])
    # y_data = copy.deepcopy(dataset.data['y'])
    # z_data = copy.deepcopy(dataset.data['z'])

    visual3d(x_data, y_data, z_data)

    t = np.linspace(0, len(x_data), len(x_data))

    figure = plt.figure(figsize=(12, 12))
    axes = figure.subplots(3, 2)
    suptitle = figure.suptitle(title)

    axes[0, 0].plot(t, x_data)
    axes[0, 0].grid()
    axes[1, 0].plot(t, y_data)
    axes[1, 0].grid()
    axes[2, 0].plot(t, z_data)
    axes[2, 0].grid()

    axes[0, 1].plot(x_data, y_data)
    axes[0, 1].grid()
    axes[1, 1].plot(x_data, z_data)
    axes[1, 1].grid()
    axes[2, 1].plot(y_data, z_data)
    axes[2, 1].grid()

    plt.show()

def to_file(model, dataset, dataset_size, input_size):
    beta1 = np.linspace(0, 0.999, num=10)
    lambda1 = np.linspace(0, 0.999, num=10)
    path = 'for_andrey.pdf'
    pdf = PdfPages(path)
    i = 0

    for beta in beta1:

        constants = {
            'alpha': 0.01,
            'lambda': 0,
            'beta': beta,
            'max_error': 0.001,
            'validation_error': 0.000001
        }
        print(str(i))
        i += 1
        response = model.learning(dataset, constants)
        result, E = model.fit(dataset.full_set, dataset_size - dataset.input_size)



        x = [row[0] for row in result]
        y = [row[1] for row in result]
        z = [row[2] for row in result]

        while len(dataset.data['x']) != len(x):
            dataset.data['x'].pop()
            dataset.data['y'].pop()
            dataset.data['z'].pop()

        # Создаем фигуру с несколькими осями.
        figure = plt.figure(figsize=(12, 12))
        axes = figure.subplots(3, 2)

        line = 'beta: {}   '.format(beta)
        suptitle = figure.suptitle(line)

        t = np.linspace(0, len(dataset.data['x']), len(dataset.data['x']))

        axes[0, 0].plot(t, dataset.data['x'], '-', t, x, ':')
        axes[0, 0].grid()
        axes[1, 0].plot(t, dataset.data['y'], '-', t, y, ':')
        axes[1, 0].grid()
        axes[2, 0].plot(t, dataset.data['z'], '-', t, z, ':')
        axes[2, 0].grid()

        val_list = model.report.square_validation_error_list
        t = np.linspace(0, len(val_list), len(val_list))
        axes[0, 1].plot(t, val_list)
        axes[0, 1].grid()

        learn_list = model.report.square_learning_error_list
        t = np.linspace(0, len(learn_list), len(learn_list))
        axes[1, 1].plot(t, learn_list)
        axes[1, 1].grid()

        test_list = model.report.square_test_error_list
        t = np.linspace(0, len(test_list), len(test_list))
        axes[2, 1].plot(t, test_list)
        axes[2, 1].grid()
        # Сохраняем страницу
        pdf.savefig(figure)

    print(path)
    pdf.close()