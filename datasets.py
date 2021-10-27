class DataSet:
    def __init__(self):
        self.learning_set = {}
        self.test_set = {}
        self.validation_set = {}

    @staticmethod
    def __data_transform(x):
        x_min = min(x)
        x_max = max(x)

        result = []
        for item in x:
            result.append((item - x_min) / (x_max - x_min))
        return result


class EquationsSystemDataset(DataSet):
    def __create_sets(self, data, size, start):
        result = {'x': [], 'e': []}
        for i in range(start, size + start):
            x = []
            e = []
            for j in range(int(self.input_size / len(data))):
                x.append(data['x'][i + j])
                x.append(data['y'][i + j])
                x.append(data['z'][i + j])
                e.append(data['x'][i + j + 1])
                e.append(data['y'][i + j + 1])
                e.append(data['z'][i + j + 1])
            result['x'].append(x)
            result['e'].append(e)
        return result

    def __init__(self, data, size, learn=0.8, test=0.2, validation=0, input_size=3):
        super().__init__()
        self.learn_size = int(size * learn)
        self.test_size = int(size * test)
        self.validation_size = int(size * validation)

        self.input_size = input_size
        self.data = data

    def create_sets(self):
        self.learning_set = self.__create_sets(self.data, self.learn_size, 0)
        self.test_set = self.__create_sets(self.data, self.test_size, self.learn_size)
        self.validation_set = self.__create_sets(self.data, self.validation_size, self.learn_size + self.test_size)

    def data_transform(self):
        self.data['x'] = self.__data_transform(self.data['x'])
        self.data['y'] = self.__data_transform(self.data['y'])
        self.data['z'] = self.__data_transform(self.data['z'])


class WindowingMethodDataSet(DataSet):
    def __init__(self, func, size, start, step=0.01, learn=0.8, test=0.2, validation=0, input_size=3):
        super().__init__()
        self.learn_size = int(size * learn)
        self.test_size = int(size * test)
        self.validation_size = int(size * validation)
        self.func = func
        self.input_size = input_size

        self.data = self.__create_data(size, start, step)

    def __create_data(self, size, start, step):
        data = []
        i = start
        for _ in range(size):
            data.append(self.func(i))
            i += step
        return data

    def __create_set(self, size, start):
        result = {'x': [], 'e': []}
        for i in range(start, size - self.input_size - start):
            x = []
            e = []
            for j in range(self.input_size):
                x.append(self.data[i + j])
            e.append(self.data[i + self.input_size])
            result['x'].append(x)
            result['e'].append(e)
        return result

    def data_transform(self):
        self.data = self.__data_transform(self.data)

    def create_sets(self):
        self.learning_set = self.__create_set(self.learn_size, 0)
        self.test_set = self.__create_set(self.test_size, self.learn_size)
        self.validation_set = self.__create_set(self.validation_size, self.learn_size + self.test_size)
