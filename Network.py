from random import uniform
import Training_Report as TR
import warnings
import copy
import Visualisation as Visual

warnings.filterwarnings('ignore')


class Perseptron():
    def __init__(self, input_layer, output_layer, functions):
        self.outputL_activation = functions[0]
        self.d_outputL_activation = functions[1]

        self.inputL = input_layer
        self.outputL = output_layer

        self.report = TR.Report()

    @staticmethod
    def error(output, e):
        error = 0
        for j in range(len(output)):
            error += (abs(output[j] - e[j])) ** 2
        error /= 2
        return error

    def validation(self, dataset, calculating):
        E = 0
        for index in range(len(dataset['x'])):
            x = dataset['x'][index]
            e = dataset['e'][index]

            calc_res = calculating(x)

            E += self.error(calc_res['output'], e)
        self.report.add_validation_error_value(E)
        return E


class SLP(Perseptron):
    def __init(self, input_layer, output_layer, functions):
        super().__init__(input_layer, output_layer, functions)

        self.W_ij = [[uniform(-1, 1) for j in range(output_layer)] for i in range(input_layer)]
        self.W_ij_past = copy.deepcopy(self.W_ij)
        self.T_j = [uniform(-1, 1) for j in range(output_layer)]

    def calculating(self, X):
        Y = []
        Sy = [0] * self.outputL

        for j in range(self.outputL):
            for i in range(self.inputL):
                Sy[j] += X[i] * self.W_ij[i][j]
            Sy[j] -= self.T_j[j]
            Y.append(self.outputL_activation(Sy[j]))

        return {'Y': Y, 'Sy': Sy}

    def modifying(self, data, X, e, constants):
        gamma_j = []
        Y = data['output']
        Sy = data['Sy']

        tmp_ij = copy.deepcopy(self.W_ij)

        for j in range(self.outputL):
            gamma_j.append(Y[j] - e[j])

        for j in range(self.outputL):
            sgd = constants['alpha'] * gamma_j[j] * self.d_outputL_activation(Sy[j])
            for i in range(self.inputL):
                weight_decay = self.W_ij[i][j] * (1 - constants['lambda'])
                heavy_ball = constants['beta'] * (self.W_ij[i][j] - self.W_ij_past[i][j])
                self.W_ij[i][j] = weight_decay - sgd * X[i] + heavy_ball
            self.T_j[j] += sgd

        self.W_ij_past = copy.deepcopy(tmp_ij)

    def learning(self, data_set, constants):
        print('Network learning...')
        dataset = data_set.learning_set
        validation_set = data_set.validation_set

        epoch = 0
        while True:
            for index in range(len(dataset['x']) - 1):
                X_train = dataset['x'][index]
                e_train = dataset['e'][index]

                calc_res = self.calculating(X_train)
                self.modifying(calc_res, X_train, e_train, constants)

                E = 0
                for i in range(len(dataset['x']) - 1):
                    X_check = dataset['x'][i]
                    e_check = dataset['e'][i]

                    calc_res = self.calculating(X_check)
                    E += self.error(calc_res['Z'], e_check)

                self.report.add_learning_error_value(E)
                if E < constants['max_error']:
                    return 'max_error'

                epoch += 1
                val_error = super().validation(validation_set, self.calculating)
                if val_error < constants['validation_error']:
                    return 'validation_error'

    def fit(self, dataset, dataset_size):
        print('Forecasting...')
        result = []
        E = 0
        for index in range(dataset_size):
            X = dataset['x'][index]
            e = dataset['e'][index]

            calc_res = self.calculating(X)
            result.append(calc_res['output'])
            e1 = self.error(calc_res['output'], e)
            self.report.add_test_error_value(e1)
            E += e1

        print('Done')
        return result, E


class MLP(Perseptron):
    def __init__(self, input_layer, hidden_layer, output_layer, functions):
        super().__init__(input_layer, output_layer, functions[1])

        self.hiddenL_activation = functions[0][0]
        self.d_hiddenL_activation = functions[0][1]

        self.hiddenL = hidden_layer

        border = 1 / (hidden_layer ** 0.5)
        self.W_ij = [[uniform(-1 * border, border) for j in range(hidden_layer)] for i in range(input_layer)]
        self.W_jk = [[uniform(-1 * border, border) for k in range(output_layer)] for j in range(hidden_layer)]
        self.W_ij_past = copy.deepcopy(self.W_ij)
        self.W_jk_past = copy.deepcopy(self.W_jk)
        self.T_j = [uniform(-1 * border, border) for j in range(hidden_layer)]
        self.T_k = [uniform(-1 * border, border) for k in range(output_layer)]

        self.report = TR.Report()

    def fit(self, dataset, dataset_size):
        print('Forecasting...')
        result = []
        E = 0
        for index in range(dataset_size):
            X = dataset['x'][index]
            e = dataset['e'][index]

            calc_res = self.calculating(X)
            result.append(calc_res['output'])
            e1 = self.error(calc_res['output'], e)
            self.report.add_test_error_value(e1)
            E += e1

        print('Done')
        return result, E

    def calculating(self, X):
        Y = []
        Sy = [0] * self.hiddenL

        Z = []
        Sz = [0] * self.outputL

        for j in range(self.hiddenL):
            for i in range(self.inputL):
                Sy[j] += X[i] * self.W_ij[i][j]
            Sy[j] -= self.T_j[j]
            Y.append(self.hiddenL_activation(Sy[j]))

        for k in range(self.outputL):
            for j in range(self.hiddenL):
                Sz[k] += Y[j] * self.W_jk[j][k]
            Sz[k] -= self.T_k[k]
            Z.append(self.outputL_activation(Sz[k]))

        return {'output': Z, 'Y': Y, 'Sy': Sy, 'S_output': Sz}

    def learning(self, data_set, constants):
        print('Network learning...')
        dataset = data_set.learning_set
        validation_set = data_set.validation_set

        epoch = 0
        while True:
            for index in range(len(dataset['x']) - 1):
                X_train = dataset['x'][index]
                e_train = dataset['e'][index]

                calc_res = self.calculating(X_train)
                self.modifying(calc_res, X_train, e_train, constants)

                E = 0
                for i in range(len(dataset['x']) - 1):
                    X_check = dataset['x'][i]
                    e_check = dataset['e'][i]

                    calc_res = self.calculating(X_check)
                    E += self.error(calc_res['output'], e_check)

                self.report.add_learning_error_value(E)
                #print(E)
                if E < constants['max_error']:
                    return 'max_error'

                epoch += 1

                val_error = self.validation(validation_set, self.calculating)
                if val_error < constants['validation_error']:
                    return 'validation_error'

    @staticmethod
    def adaptive_alpha(constants, data):
        if constants['output alpha'] == 0:
            temp = 0
            for item in data['Y']:
                temp += item ** 2
            constants['output alpha'] = temp

    def modifying(self, data, X, e, constants):
        gamma_k = []
        gamma_j = [0] * self.hiddenL

        # self.adaptive_alpha(constants, data)

        Z = data['output']
        Y = data['Y']
        Sy = data['Sy']
        Sz = data['S_output']

        tmp_ij = copy.deepcopy(self.W_ij)
        tmp_jk = copy.deepcopy(self.W_jk)

        for k in range(self.outputL):
            gamma_k.append(Z[k] - e[k])

        for j in range(self.hiddenL):
            for k in range(self.outputL):
                gamma_j[j] += gamma_k[k] * self.d_outputL_activation(Sy[j]) * self.W_jk[j][k]

        for k in range(self.outputL):
            sgd = constants['alpha'] * gamma_k[k] * self.d_outputL_activation(Sz[k])
            for j in range(self.hiddenL):
                weight_decay = self.W_jk[j][k] * (1 - constants['lambda'])
                heavy_ball = constants['beta'] * (self.W_jk[j][k] - self.W_jk_past[j][k])
                self.W_jk[j][k] = weight_decay - sgd * Y[j] + heavy_ball
            self.T_k[k] += sgd

        for j in range(self.hiddenL):
            sgd = constants['alpha'] * gamma_j[j] * self.d_hiddenL_activation(Sy[j])
            for i in range(self.inputL):
                weight_decay = self.W_ij[i][j] * (1 - constants['lambda'])
                heavy_ball = constants['beta'] * (self.W_ij[i][j] - self.W_ij_past[i][j])
                self.W_ij[i][j] = weight_decay - sgd * X[i] + heavy_ball
            self.T_j[j] += sgd

        self.W_jk_past = copy.deepcopy(tmp_jk)
        self.W_ij_past = copy.deepcopy(tmp_ij)

    def first_step_modifying(self, e, Yj, s, Z, constants):
        tmp_jk = copy.deepcopy(self.W_jk)
        for k in range(self.outputL):
            sgd = constants['output alpha'] * (Z[k] - e[k]) * self.d_outputL_activation(s[k])
            for j in range(self.hiddenL):
                weight_decay = self.W_jk[j][k] * (1 - constants['lambda'])
                heavy_ball = constants['beta'] * (self.W_jk[j][k] - self.W_jk_past[j][k])
                self.W_jk[j][k] = weight_decay - sgd * Yj[j] + heavy_ball
            self.T_k[k] += sgd
        self.W_jk_past = copy.deepcopy(tmp_jk)

    def second_step_modifying(self, x, constants, s, hs):
        gamma = []
        for j in range(self.hiddenL):
            gamma.append(self.hiddenL_activation(s[j]) - self.hiddenL_activation(hs[j]))
            # gamma.append(s[j] - hs[j])

        tmp_ij = copy.deepcopy(self.W_ij)
        for j in range(self.hiddenL):
            sgd = constants['hidden alpha'] * gamma[j] * self.d_hiddenL_activation(s[j])
            # sgd = constants['hidden alpha'] * gamma[j] * s[j]
            for i in range(self.inputL):
                weight_decay = self.W_ij[i][j] * (1 - constants['hidden lambda'])
                heavy_ball = constants['hidden beta'] * (self.W_ij[i][j] - self.W_ij_past[i][j])
                self.W_ij[i][j] = weight_decay - sgd * x[i] + heavy_ball
            self.T_j[j] += sgd
        self.W_ij_past = copy.deepcopy(tmp_ij)

    def first_step_calc(self, sum1):
        y = [self.hiddenL_activation(sum1[k]) for k in range(self.hiddenL)]
        s = [0] * self.outputL
        z = []
        for k in range(self.outputL):
            for j in range(self.hiddenL):
                s[k] += y[j] * self.W_jk[j][k]
            s[k] -= self.T_k[k]
            z.append(self.outputL_activation(s[k]))
        return z, y

    def second_step_calc(self, x):
        y = []
        s = [0] * self.hiddenL
        for j in range(self.hiddenL):
            for i in range(self.inputL):
                s[j] += x[i] * self.W_ij[i][j]
            s[j] -= self.T_j[j]
            y.append(self.hiddenL_activation(s[j]))
        return s, y

    def first_step_learning(self, h_sum, dataset, constants):
        learning_set = dataset.learning_set
        err_list = []
        ind = 0
        while True:
            for i in range(dataset.learn_size):
                e = learning_set['e'][i]
                z, y = self.first_step_calc(h_sum[i])
                self.first_step_modifying(e, y, h_sum[i], z, constants)

                error = 0
                for index in range(dataset.learn_size):
                    z_check, y = self.first_step_calc(h_sum[index])
                    e_check = learning_set['e'][index]
                    error += self.error(z_check, e_check)
                print(error)
                err_list.append(error)
                ind += 1
                if error < constants['max_first_step_error']:
                    return err_list

    @staticmethod
    def generate_sum(size, l_border=-1, r_border=1):
        s = [uniform(l_border, r_border) for _ in range(size)]
        return s

    def second_step_learning(self, dataset, constants, h_sum):
        learning_set = dataset.learning_set
        err_list = []
        ind = 0
        counter = 0
        while True:
            for i in range(dataset.learn_size):
                x = learning_set['x'][i]
                s, y = self.second_step_calc(x)
                self.second_step_modifying(x, constants, s, h_sum[i])

                error = 0
                for index in range(dataset.learn_size):
                    x_check = learning_set['x'][index]
                    s, y_check = self.second_step_calc(x_check)
                    e_check = [self.hiddenL_activation(item) for item in h_sum[index]]
                    error += self.error(y_check, e_check)
                print(error)
                err_list.append(error)
                ind += 1
                if error < constants['max_first_step_error']:
                    return err_list

                counter += 1

                if counter > 330:
                    flag = False
                    for i_check in range(1, 300):
                        last = err_list[-1 * i_check]
                        pre_last = err_list[-1 * (i_check + 1)]
                        tmp = abs(last - pre_last)
                        if tmp > 0.005:
                            flag = False
                            break
                        else:
                            flag = True

                    if flag:
                        return err_list

    def second_step_learning(self, h_sum, dataset, constants):
        learning_set = dataset.learning_set
        err_list = []
        counter = 0
        while True:
            for index in range(dataset.learn_size):
                x = learning_set['x'][index]
                s = [0] * self.hiddenL
                for j in range(self.hiddenL):
                    for i in range(self.inputL):
                        s[j] += x[i] * self.W_ij[i][j]
                    s[j] -= self.T_j[j]

                self.second_step_modifying(x, constants, s, h_sum[index])

                error = 0
                for ind in range(dataset.learn_size):
                    x = learning_set['x'][ind]
                    s_test = [0] * self.hiddenL
                    for j in range(self.hiddenL):
                        for i in range(self.inputL):
                            s_test[j] += x[i] * self.W_ij[i][j]
                        s_test[j] -= self.T_j[j]
                    error += self.error(s_test, h_sum[ind])

                print(str(error) + '      ' + str(counter))
                err_list.append(error)
                counter += 1
                if error < constants['max_second_step_error']:
                    return err_list

                if counter > 130:
                    flag = False
                    for i_check in range(1, 100):
                        last = err_list[-1 * i_check]
                        pre_last = err_list[-1 * (i_check + 1)]
                        tmp = abs(last - pre_last)
                        if tmp > 0.1:
                            flag = False
                            break
                        else:
                            flag = True

                    if flag:
                        return err_list

    def step_by_step_training(self, data_set, constants):
        h_sum = [self.generate_sum(self.hiddenL) for _ in range(data_set.learn_size)]
        print('Step#1')
        l = self.first_step_learning(h_sum, data_set, constants)
        print('Step#2')
        Visual.visual2d(l)
        l1 = self.second_step_learning(data_set, constants, h_sum)

        Visual.visual2d(l1)


