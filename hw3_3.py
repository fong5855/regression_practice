import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


initial_learning_rate = 1e-9
iterations = 100000

# data
x_list = [1, 1.6, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 9.9]
y_list = [27, 32.5, 30, 37.3, 36.4, 32.4, 28.5, 30, 34.1, 39, 36]
x_data = np.asarray(x_list, dtype=np.float32)
y_data = np.asarray(y_list, dtype=np.float32)
n_samples = x_data.shape[0]


def func_5th(a, b, c, d, e, f):
    return a * x_data * x_data * x_data * x_data * x_data \
           + b * x_data * x_data * x_data * x_data \
           + c * x_data * x_data * x_data \
           + d * x_data * x_data \
           + e * x_data \
           + f


def quard_fitting_5th():
    global_step = tf.Variable(0, trainable=False)
    add_global = global_step.assign_add(1)

    a = tf.Variable(tf.random_uniform([1], 0, 0.01))
    b = tf.Variable(tf.random_uniform([1], -0.01, 0.01))
    c = tf.Variable(tf.random_uniform([1], -0.01, 0.01))
    d = tf.Variable(tf.random_uniform([1], -0.01, 0.01))
    e = tf.Variable(tf.random_uniform([1], -0.01, 0.01))
    f = tf.Variable(tf.random_uniform([1], 27, 29))

    y = func_5th(a, b, c, d, e, f)
    loss = tf.reduce_sum(tf.pow(y-y_data, 2) / (2*n_samples))

    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=10000, decay_rate=0.95)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # training
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for step in range(iterations+1):
        sess.run(train)
        sess.run([add_global, learning_rate])
        if step % 100 == 0:
            print(step, sess.run(a), sess.run(b), sess.run(c),
                  sess.run(d), sess.run(e), sess.run(f), sess.run(loss))

    return sess.run(a), sess.run(b), sess.run(c), sess.run(d), sess.run(e), sess.run(f)


if __name__ == '__main__':
    a, b, c, d, e, f = quard_fitting_5th()

    plt.plot(x_data, y_data, 'ro', label='origin data')
    plt.plot(func_5th(a, b, c, d, e, f), label='5th order fitted line')
    plt.legend()
    plt.show()
