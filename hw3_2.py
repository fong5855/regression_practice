import numpy as np
from math import sqrt
phi = (1 + sqrt(5))/2
resphi = 2 - phi


# data
x_list = [1, 1.6, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 9.9]
y_list = [27, 32.5, 30, 37.3, 36.4, 32.4, 28.5, 30, 34.1, 39, 36]
x_data = np.asarray(x_list, dtype=np.float32)
y_data = np.asarray(y_list, dtype=np.float32)
n_samples = x_data.shape[0]


def golden_section(func, a, b):
    if abs(a-b) < 1e-3:
        return (a+b)/2

    c =


def line_function(a0, a1):
    f = 0
    for i in range(n_samples):
        f += (y_data[i] - (a0 + a1 * x_data[i]))**2
    return f


def grad_line_function(a0, a1):
    f1 = 0
    for i in range(n_samples):
        f1 += -2*y_data[i] + 2*a0 + 2*a1*x_data[i]
    f2 = 0
    for i in range(n_samples):
        f2 += -2*y_data[i]*x_data[i] + 2*a0*x_data[i] + 2*a1*(x_data[i])**2
    return f1, f2


def line_conjugate(x0, y0):
    # grad
    f1, f2 = grad_line_function(0, 0)







if __name__ == '__main__':
    line_conjugate(0, 0)