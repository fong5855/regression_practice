import numpy as np
import matplotlib.pyplot as plt

# data
x_list = [1, 1.6, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 9.9]
y_list = [27, 32.5, 30, 37.3, 36.4, 32.4, 28.5, 30, 34.1, 39, 36]
x_data = np.asarray(x_list, dtype=np.float32)
y_data = np.asarray(y_list, dtype=np.float32)
n_samples = x_data.shape[0]


def line():
    b = (n_samples * np.sum(x_data * y_data)
         - np.sum(x_data) * np.sum(y_data)) / (
            n_samples * np.sum(np.square(x_data)) - np.square(np.sum(x_data)))
    a = (np.sum(y_data) - b * np.sum(x_data)) / n_samples

    print('a=', a)
    print('b=', b)
    return b, a


def quad():
    sxx = np.sum(np.square(x_data)) / n_samples - np.mean(x_data)**2
    sxy = np.sum(x_data*y_data) / n_samples - np.mean(x_data) * np.mean(y_data)
    sxx2 = np.sum(np.power(x_data, 3)) / n_samples - np.mean(x_data) * np.mean(np.square(x_data))
    sx2x2 = np.sum(np.power(x_data, 4)) / n_samples - np.mean(np.square(x_data)) * np.mean(np.square(x_data))
    sx2y = np.sum(np.square(x_data)*y_data) / n_samples - np.mean(np.square(x_data)) * np.mean(y_data)

    c = (sx2y * sxx - sxy * sxx2) / (sxx * sx2x2 - sxx2**2)
    b = (sxy * sx2x2 - sx2y * sxx2) / (sxx * sx2x2 - sxx2**2)
    a = np.mean(y_data) - b * np.mean(x_data) - c * np.mean(np.square(x_data))

    print('A=', a)
    print('B=', b)
    print('C=', c)
    return c, b, a


if __name__ == '__main__':
    lb, la = line()
    qc, qb, qa = quad()

    plt.plot(x_data, y_data, 'ro', label='origin data')
    plt.plot(x_data, lb * x_data + la, label='linear fitting')
    plt.plot(x_data, qc * x_data * x_data + qb * x_data + qa, label='quadratic fitting')
    plt.legend()
    plt.show()
