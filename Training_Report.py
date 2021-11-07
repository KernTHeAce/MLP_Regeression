from Visualisation import visual2d


class Report:
    def __init__(self):
        self.square_learning_error_list = []
        self.average_learning_error = 0

        self.square_test_error_list = []
        self.average_test_error = 0

        self.square_validation_error_list = []
        self.average_validation_error = 0

    def add_learning_error_value(self, value):
        self.square_learning_error_list.append(value)
        self.average_learning_error += value

    def add_test_error_value(self, value):
        self.square_test_error_list.append(value)
        self.average_learning_error += value

    def add_validation_error_value(self, value):
        self.square_validation_error_list.append(value)
        self.average_validation_error += value

    def get_learning_average_error(self):
        return self.average_learning_error / len(self.square_learning_error_list)

    def get_test_average_error(self):
        return self.average_test_error / len(self.square_test_error_list)

    def get_validation_average_error(self):
        return self.average_validation_error / len(self.square_validation_error_list)

    def learning_error_visual(self):
        visual2d(self.square_learning_error_list, 'error')

    def validation_error_visual(self):
        visual2d(self.square_validation_error_list, 'error')

    def test_error_visual(self):
        visual2d(self.square_test_error_list, 'error')