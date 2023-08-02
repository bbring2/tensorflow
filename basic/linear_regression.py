import tensorflow as tf
import numpy as np

x_train = np.array([1, 2, 3, 4, 5], dtype=np.float32)
y_train = np.array([2, 4, 6, 8, 10], dtype=np.float32)

# 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])  # input : 1D
])

# compile
model.compile(optimizer='sgd', loss='mean_squared_error')

# 모델 학습
model.fit(x_train, y_train, epochs=1000)

# 예측
x_test = np.array([8, 7], dtype=np.float32)
y_pred = model.predict(x_test)

print(y_pred)

