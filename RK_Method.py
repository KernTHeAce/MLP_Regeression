from Training_Report import data_transform

class EquationsSystem:
    def __init__(self, dx_function, dy_function, dz_function):

        self.dx = dx_function
        self.dy = dy_function
        self.dz = dz_function

    def calculating(self, num, h):
        x = [0.1]
        y = [0.1]
        z = [0.1]

        k = [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        print('DataSet creating...')
        for i in range(num-1):
            h2 = h / 2
            k[0][0] = self.dx(x[i], y[i], z[i])
            k[1][0] = self.dy(x[i], y[i], z[i])
            k[2][0] = self.dz(x[i], y[i], z[i])

            k[0][1] = self.dx(x[i] + k[0][0] * h2, y[i] + k[1][0] * h2, z[i] + k[2][0] * h2)
            k[1][1] = self.dy(x[i] + k[0][0] * h2, y[i] + k[1][0] * h2, z[i] + k[2][0] * h2)
            k[2][1] = self.dz(x[i] + k[0][0] * h2, y[i] + k[1][0] * h2, z[i] + k[2][0] * h2)

            k[0][2] = self.dx(x[i] + k[0][1] * h2, y[i] + k[1][1] * h2, z[i] + k[2][1] * h2)
            k[1][2] = self.dy(x[i] + k[0][1] * h2, y[i] + k[1][1] * h2, z[i] + k[2][1] * h2)
            k[2][2] = self.dz(x[i] + k[0][1] * h2, y[i] + k[1][1] * h2, z[i] + k[2][1] * h2)

            k[0][3] = self.dx(x[i] + k[0][2] * h, y[i] + k[1][2] * h, z[i] + k[2][2] * h)
            k[1][3] = self.dy(x[i] + k[0][2] * h, y[i] + k[1][2] * h, z[i] + k[2][2] * h)
            k[2][3] = self.dz(x[i] + k[0][2] * h, y[i] + k[1][2] * h, z[i] + k[2][2] * h)

            x_tmp = x[i] + (k[0][0] + 2 * k[0][1] + 2 * k[0][2] + k[0][3]) * (h / 6)
            y_tmp = y[i] + (k[1][0] + 2 * k[1][1] + 2 * k[1][2] + k[1][3]) * (h / 6)
            z_tmp = z[i] + (k[2][0] + 2 * k[2][1] + 2 * k[2][2] + k[2][3]) * (h / 6)

            x.append(x_tmp)
            y.append(y_tmp)
            z.append(z_tmp)

        print('Done')
        return {'x': data_transform(x), 'y': data_transform(y), 'z': data_transform(z)}


