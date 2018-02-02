import tensorflow as tf
import matplotlib.pyplot as plt

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b

y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

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

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


#========================
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_train, y_train, c="r")
plt.show()


