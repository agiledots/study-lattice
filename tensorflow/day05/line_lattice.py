# https://qiita.com/taigamikami/items/6c69fc813940f838e96c

import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl
import matplotlib.pyplot as plt
import input_data

# ====================================
# 訓練用のデータ
# ====================================
#x_train = np.arange(-5, 5, 0.2)
#noise = np.random.normal(0, 4, x_train.shape)
#y_train = np.square(x_train) + noise

data = input_data.read_data("train")
x_train = data.T[0]
y_train = data.T[1]

batch_size = len(x_train)

# input_fn = tf.estimator.inputs.numpy_input_fn(
#     {"x": x_train}, y_train, batch_size=batch_size, num_epochs=None, shuffle=True)

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
num_keypoints = 10
# hparams = tfl.CalibratedRtlHParams(
#     num_keypoints=num_keypoints,
#     num_lattices=5,
#     lattice_rank=2,
#     learning_rate=0.01)

hparams = tfl.CalibratedLinearHParams(
    num_keypoints=num_keypoints,
    num_lattices=10,
#    lattice_rank=2,
    learning_rate=0.1)

# Set feature monotonicity.
#hparams.set_feature_param('x', 'monotonicity', -1)

# Define keypoint init.
keypoints_init_fns = {
    'x': lambda: tfl.uniform_keypoints_for_signal(num_keypoints,
                                                         input_min=-5.0,
                                                         input_max=5.0,
                                                         output_min=0.0,
                                                         output_max=25.0),
}

print("keypoints_init_fns: %r" % keypoints_init_fns)

# ====================================
# 訓練
# ====================================
# lattice_estimator = tfl.calibrated_lattice_regressor(
#     feature_columns=feature_columns,
#     hparams=hparams,
#     keypoints_initializers_fn=keypoints_init_fns
# )
lattice_estimator = tfl.calibrated_linear_regressor(
    feature_columns=feature_columns,
    hparams=hparams,
    keypoints_initializers_fn=keypoints_init_fns
)

# Train!
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_train},
    y=y_train,
    batch_size=batch_size,
    num_epochs=1000,
    shuffle=False)

train_metrics = lattice_estimator.train(input_fn=train_input_fn)

# ====================================
# モデルの評価
# ====================================
eval_metrics = lattice_estimator.evaluate(input_fn=train_input_fn)
print("train metrics: %r"% eval_metrics)


# ====================================
# 検証用データ
# ====================================
eval_data = input_data.read_data("eval")
x_eval = eval_data.T[0]
y_eval = eval_data.T[1]
#
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

eval_metrics = lattice_estimator.evaluate(input_fn=eval_input_fn)
print("eval metrics: %r"% eval_metrics)


# ====================================
# 予測
# ====================================
predict_data = input_data.read_data("predict")
x_predict = predict_data.T[0]

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_predict},
    y=None,
    batch_size=batch_size,
    num_epochs=1,
    shuffle=False
)
predict_results = list(lattice_estimator.predict(input_fn=predict_input_fn))


# ====================================
# データを図表に表示する
# ====================================
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(x_train, y_train)

y_predict = np.array([])
for prediction in predict_results:
    y_predict = np.append(y_predict, prediction["predictions"][0])

ax1.plot(x_eval, y_predict, "r-")
plt.show()
