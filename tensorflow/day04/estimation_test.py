import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = tf.divide(W, x) + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
# x_train = [0.5, 1, 2, 3, 4, 5]
# y_train = [2, 1, 0.5, 0.3, 0.25, 0.2]

x_train = np.linspace(0.0000001, 10, 6)
noise = np.random.normal(0, 0.1, x_train.shape)
y_train = np.divide(np.ones(x_train.shape, np.float32),  x_train) #+ noise

print("x_train: %s" % str(x_train))
print("y_train: %s" % str(y_train))

#========================
fig = plt.figure()
#========================

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

  # 学習の過程を見たいので追加
  if i%100 == 0:
      curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
      print("%d回 W: %s b: %s loss: %s"%(i,curr_W, curr_b, curr_loss))

      #=====================
      # predition_value = sess.run(y, feed_dict={x_train: x_train})
      # try:
      #     ax.lines.remove(lines[0])
      # except Exception:
      #     pass
      #
      # lines = ax.plot(x_train, predition_value, 'r-', lw=5)
      # plt.pause(0.2)

      # =====================



# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


ax = fig.add_subplot(111)
ax.scatter(x_train, y_train, c="r")

ax.scatter(x_train, sess.run(W) / x_train + sess.run(b), c="b")
#plt.plot(x_train, sess.run(W) / x_train + sess.run(b), label = "Fitted Line")

plt.show()

