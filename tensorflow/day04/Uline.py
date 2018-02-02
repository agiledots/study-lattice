import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


# ==========================
# 机器学习用数据
# ==========================
x_data = np.linspace(-1, 1, 50)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x_data.shape)
y_data = np.square(x_data) + noise


# 变量
ys = tf.placeholder(tf.float32, [None, 1])
xs = tf.placeholder(tf.float32, [None, 1])

#
layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.elu)


# 予測
predition = add_layer(layer1, 10, 1, activation_function=None)


loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition), reduction_indices=[1]))

# 优化器
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # 図表
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)

    # y labelの範囲
    # ax.set_ylim([-1.5, 1.5])

    plt.ion()
    plt.show()


    for i in range(1000):
        # 机器学习
        sess.run(train_step, feed_dict={xs: x_data , ys: y_data})

        if i % 50 == 0:
            result = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
            print(result)

            predition_value = sess.run(predition, feed_dict={xs:x_data})
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass

            #lines = ax.plot(x_data, predition_value, 'r-', lw=5)
            lines = ax.plot(x_data, predition_value, 'r')
            plt.pause(0.2)

    plt.pause(interval=3600)

