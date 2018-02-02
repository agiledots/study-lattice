import tensorflow as tf
import numpy as np

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.restore(sess, "model.ckpt")





