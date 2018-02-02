import tensorflow as tf
# NumPyは、データのロード、操作、および前処理によく使用されます。
import numpy as np

# 機能のリストを宣言する。 1つの数値機能しかありません。 より複雑で有用な他の多くのタイプの列があります。
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# 推定子は、トレーニング（フィッティング）と評価（推論）を呼び出すためのフロントエンドです。
# 線形回帰、線形分類、多くのニューラルネットワーク分類器および回帰子のような多くの事前定義型があります。
# 次のコードは、線形回帰を行う推定器を提供します。
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlowには、データセットを読み込んで設定するための多くのヘルパーメソッドが用意されています。
# ここでは、トレーニング用と評価用の2つのデータセットを使用します
# 私たちが望むデータのバッチ数（num_epochs）と各バッチの大きさを関数に伝えなければなりません。
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# このメソッドを呼び出してトレーニングデータセットを渡すことで、1000のトレーニングステップを呼び出すことができます。
estimator.train(input_fn=input_fn, steps=1000)

# モデルの評価
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
