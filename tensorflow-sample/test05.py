import tensorflow as tf

# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
biases = tf.Variable(tf.zeros([200]), name="biases")

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, when launching the model
with tf.Session() as sess:
  # Run the init operation.
  sess.run(init_op)

  save_path = saver.save(sess, "./model.ckpt")
  print("Model saved in file: ", save_path)


