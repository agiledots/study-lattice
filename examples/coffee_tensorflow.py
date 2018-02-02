import tensorflow as tf
import numpy as np

# ====================================
# Example training and testing data.
# ====================================
train_features = {
    'distance': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    'quality': np.array([2.0, 5.0, 1.0, 2.0, 5.0]),
}
train_labels = np.array([0.2, 1.0, 0.0, 0.0, 1.0])

# ====================================
# Same quality but different distance.
# ====================================
test_features = {
    'distance': np.array([5.0, 10.0]),
    'quality': np.array([3.0, 3.0]),
}

# ====================================
# Feature definition
# ====================================
feature_columns = [
    tf.feature_column.numeric_column('distance'),
    tf.feature_column.numeric_column('quality'),
]

input_fn = tf.estimator.inputs.numpy_input_fn(
    train_features, train_labels, batch_size=5, num_epochs=None, shuffle=True)


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    train_features, train_labels, batch_size=5, num_epochs=1000, shuffle=False)


# ====================================
# 訓練(トレーニング)
# ====================================
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
estimator.train(input_fn=input_fn, steps=1000)


# ====================================
# モデルの評価
# ====================================
train_metrics = estimator.evaluate(input_fn=train_input_fn)
print("train metrics: %r"% train_metrics)



# ====================================
# 予測
# ====================================
predict_fn = tf.estimator.inputs.numpy_input_fn(
    test_features, None, batch_size=5, shuffle=False)

predict_results = list(estimator.predict(input_fn=predict_fn))
print("predict results: %r " % predict_results)















