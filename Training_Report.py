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






