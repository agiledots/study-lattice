# https://qiita.com/taigamikami/items/6c69fc813940f838e96c

import tensorflow as tf
# NumPyは、データのロード、操作、および前処理によく使用されます。
import numpy as np
import matplotlib.pyplot as plt


# 機能のリストを宣言する。 1つの数値機能しかありません。 より複雑で有用な他の多くのタイプの列があります。
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# 推定子は、トレーニング（フィッティング）と評価（推論）を呼び出すためのフロントエンドです。
# 線形回帰、線形分類、多くのニューラルネットワーク分類器および回帰子のような多くの事前定義型があります。
# 次のコードは、線形回帰を行う推定器を提供します。
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
#estimator = tf.estimator.DNNLinearCombinedRegressor(linear_feature_columns=feature_columns)

# TensorFlowには、データセットを読み込んで設定するための多くのヘルパーメソッドが用意されています。
# ここでは、トレーニング用と評価用の2つのデータセットを使用します
# 私たちが望むデータのバッチ数（num_epochs）と各バッチの大きさを関数に伝えなければなりません。
#x_train = np.array([1., 2., 3., 4.])
#y_train = np.array([0., -1., -2., -3.])


# ====================================
# 訓練用のデータ
# ====================================
# start:1  end:10  step=:0.2
x_train = np.arange(1, 10, 0.2)

# 斯分布（Gaussian Distribution）的概率密度函数（probability density function）
# numpy.random.normal(loc=0.0, scale=1.0, size=None)
#
# 参数的意义为：
# loc：float
#     此概率分布的均值（对应着整个分布的中心centre）
# scale：float
#     此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
# size：int or tuple of ints
#     输出的shape，默认为None，只输出一个值

noise = np.random.normal(0, 0.5, x_train.shape)
y_train = x_train + noise


# ====================================
# 検証用データ
# ====================================
x_eval = np.arange(1, 10, 0.2)
noise = np.random.normal(0, 0.5, x_train.shape)
y_eval = x_train + noise


# x_train = np.array([1., 2., 3., 4.])
# y_train = np.array([0., -1., -2., -3.])

batch_size = len(x_train)
print("batch_size: %d"  % batch_size)

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=batch_size, num_epochs=None, shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=batch_size, num_epochs=1000, shuffle=False)

# ====================================
# 訓練(トレーニング)
# ====================================
# このメソッドを呼び出してトレーニングデータセットを渡すことで、
# 1000のトレーニングステップを呼び出すことができます。
estimator.train(input_fn=input_fn, steps=1000)

# ====================================
# モデルの評価
# ====================================
train_metrics = estimator.evaluate(input_fn=train_input_fn)
print("train metrics: %r"% train_metrics)

# ====================================
# 検証用データ評価
# ====================================
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("eval metrics: %r"% eval_metrics)



# ====================================
# 予測
# ====================================
x_predict = np.arange(1, 20, 0.1)
predict_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_predict}, None, batch_size=batch_size, shuffle=False)

predict_results = list(estimator.predict(input_fn=predict_fn))
print(predict_results)

y_predict = np.array([])
for prediction in predict_results:
    print(prediction["predictions"])
    y_predict = np.append(y_predict, prediction["predictions"][0])

print(y_predict)

# ====================================
# データを図表に表示する
# ====================================
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(x_train, y_train)
ax1.scatter(x_eval, y_eval)
ax1.scatter(x_predict, y_predict)

plt.show()
