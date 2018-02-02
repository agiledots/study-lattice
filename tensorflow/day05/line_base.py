# https://qiita.com/taigamikami/items/6c69fc813940f838e96c

import tensorflow as tf
# NumPyは、データのロード、操作、および前処理によく使用されます。
import numpy as np
import matplotlib.pyplot as plt
import input_data

# ====================================
# 訓練用のデータ
# ====================================
data = input_data.read_data("train")
x_train = data.T[0]
y_train = data.T[1]

#
batch_size = len(x_train)
print("batch_size: %d"  % batch_size)

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=batch_size, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=batch_size, num_epochs=1000, shuffle=False)


# ====================================
# Model 
# ====================================
# 機能のリストを宣言する。 1つの数値機能しかありません。 より複雑で有用な他の多くのタイプの列があります。
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

#
#estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
#estimator = tf.estimator.DNNLinearCombinedRegressor(linear_feature_columns=feature_columns)
# http://d0evi1.com/tensorflow/estimators/
estimator = tf.estimator.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[10, 10, 10],
        #activation_fn=tf.nn.relu,
    )

# ====================================
# 訓練
# ====================================
estimator.train(input_fn=input_fn, steps=1000)

# ====================================
# モデルの評価
# ====================================
train_metrics = estimator.evaluate(input_fn=train_input_fn)
print("train metrics: %r"% train_metrics)


# ====================================
# 検証用データ
# ====================================
# x_eval = np.arange(-5, 5, 0.5)
# noise = np.random.normal(0, 1, x_eval.shape)
# y_eval = np.square(x_eval) + noise

eval_data = input_data.read_data("eval")
x_eval = eval_data.T[0]
y_eval = eval_data.T[1]

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("eval metrics: %r"% eval_metrics)


# ====================================
# 予測
# ====================================
predict_data = input_data.read_data("predict")
x_predict = predict_data.T[0]

predict_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_predict}, None, batch_size=batch_size, shuffle=False)

predict_results = list(estimator.predict(input_fn=predict_fn))

# 予測結果整理
y_predict = np.array([])
for prediction in predict_results:
    y_predict = np.append(y_predict, prediction["predictions"][0])


# ====================================
# データを図表に表示する
# ====================================
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(x_train, y_train)

ax1.plot(x_train, y_predict, "r-")
ax1.scatter(x_train, y_predict)

plt.show()
