from Visualisation import visual2d

class Report:
    def __init__(self):
        self.epoch = 0
        self.square_learning_error_list = []
        self.average_learning_error = 0

        self.square_test_error_list = []
        self.average_test_error = 0

    def count(self):
        self.epoch += 1

    def add_learning_error_value(self, value):
        self.square_learning_error_list.append(value)
        self.average_learning_error += value

    def add_test_error_value(self, value):
        self.square_learning_error_list.append(value)
        self.average_learning_error += value

    def get_learning_average_error(self):
        return self.average_learning_error / len(self.square_learning_error_list)

    def error_visual(self):
        visual2d(self.square_learning_error_list, 'error')


class MLPDataSet:
    @staticmethod
    def create_sets(data, size, start):
        ds = []
        for i in range(start, size + start):
            x = data['x'][i]
            y = data['y'][i]
            z = data['z'][i]
            ds.append([x, y, z])
        return ds

    def __init__(self, data):
        size = len(data['x'])
        learning_size = int(0.8 * size)
        validation_size = int(0.1 * size)
        test_size = int(0.1 * size)

        self.test_set = self.create_sets(data, test_size, 0)
        self.learning_set = self.create_sets(data, learning_size, test_size)
        self.validation_set = self.create_sets(data, validation_size, learning_size + test_size)


    def get_ls_len(self):
        return len(self.learning_set)

    def get_vs_len(self):
        return len(self.validation_set)

    def get_ts_len(self):
        return len(self.test_set)


def data_transform(x):
    x_min = min(x)
    x_max = max(x)

    result = []
    for item in x:
        result.append((item - x_min) / (x_max - x_min))
    return result



