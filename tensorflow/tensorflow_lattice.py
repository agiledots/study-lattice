# https://qiita.com/taigamikami/items/6c69fc813940f838e96c

import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl
import matplotlib.pyplot as plt


# ====================================
# 訓練用のデータ
# ====================================
x_train = np.arange(-5, 5, 0.2)
noise = np.random.normal(0, 2, x_train.shape)
y_train = np.square(x_train) + noise

batch_size = len(x_train)

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=batch_size, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=batch_size, num_epochs=1000, shuffle=False)

# ====================================
# 訓練(トレーニング)
# ====================================

# 機能のリストを宣言する。 1つの数値機能しかありません。 より複雑で有用な他の多くのタイプの列があります。
feature_columns = [
    tf.feature_column.numeric_column("x")
]

# Hyperparameters.
num_keypoints = 20
hparams = tfl.CalibratedLatticeHParams(
    feature_names=['x'],
    num_keypoints=num_keypoints,
    learning_rate=0.1,
    lattice_rank=2
)

# Set feature monotonicity.
hparams.set_feature_param('x', 'monotonicity', -1)

# Define keypoint init.
keypoints_init_fns = {
    'x': lambda: tfl.uniform_keypoints_for_signal(num_keypoints,
                                                         input_min=-5.0,
                                                         input_max=5.0,
                                                         output_min=0.0,
                                                         output_max=25.0),
}

lattice_estimator = tfl.calibrated_lattice_regressor(
    feature_columns=feature_columns,
    hparams=hparams,
    keypoints_initializers_fn=keypoints_init_fns)


# Train!
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_train},
    y=y_train,
    batch_size=1,
    num_epochs=1000,
    shuffle=False)

lattice_estimator.train(input_fn=train_input_fn)


# ====================================
# モデルの評価
# ====================================
train_metrics = lattice_estimator.evaluate(input_fn=train_input_fn)
print("train metrics: %r"% train_metrics)




# ====================================
# 検証用データ
# ====================================
# array([-5. , -4.5, -4. , -3.5, -3. , -2.5, -2. , -1.5, -1. , -0.5,  0. , 0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5])
x_eval = np.arange(-5, 5, 0.5)
print("x_eval: %r" %x_eval)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_eval},
    y=None,
    batch_size=1,
    num_epochs=1,
    shuffle=False)

predict_results = list(lattice_estimator.predict(input_fn=test_input_fn))

print("predict: %r " % predict_results)

exit()

# # ====================================
# # データを図表に表示する
# # ====================================
# fig = plt.figure()
# ax1 = fig.add_subplot(1, 1, 1)
# ax1.scatter(x_train, y_train)

# y_predict = np.array([])
# for prediction in predict_results:
#     y_predict = np.append(y_predict, prediction["predictions"][0])

# print("x_train: %r" %x_train)
# print("y_predict: %r" %y_predict)

# ax1.plot(x_eval, y_predict, "r-")

# plt.show()
