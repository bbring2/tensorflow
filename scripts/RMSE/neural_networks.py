import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

# 현재 스크립트의 디렉토리 경로 가져오기
input_file_path = 'D:/data-science/tensorflow/datasets/kernelinput.csv'
output_file_path = 'D:/data-science/tensorflow/datasets/centralProbability.csv'

input_data = pd.read_csv(input_file_path)
output_data = pd.read_csv(output_file_path)
data = pd.concat([input_data, output_data], axis=1)

X = input_data.values
y = output_data.values
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
y_normalized = scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)


# 하이퍼파라미터 조정을 위한 함수
def create_model(hidden_units=128, activation='relu', output_activation=None, optimizer='adam'):
    model = Sequential()
    model.add(Dense(hidden_units, activation=activation, input_shape=(X_train.shape[1],)))
    if output_activation is not None:  # 출력층의 활성화 함수가 None이 아닐 때만 추가
        model.add(Dense(1, activation=output_activation))
    else:
        model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

hidden_units_list = [64, 128]
activation_list = ['relu', 'tanh']
output_activation_list = [None]  # 회귀 문제이므로 출력층의 활성화 함수는 None으로 유지
optimizer_list = ['adam', 'RMSprop']

best_rmse = float('inf')
best_model = None

for hidden_units in hidden_units_list:
    for activation in activation_list:
        for optimizer in optimizer_list:
            # output_activation_list를 반복문에서 사용하면서 하나의 활성화 함수만 선택해야 합니다.
            for output_activation in output_activation_list:
                model = create_model(hidden_units=hidden_units, activation=activation, output_activation=output_activation, optimizer=optimizer)
                model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
                y_pred_normalized = model.predict(X_test)
                y_pred = scaler.inverse_transform(y_pred_normalized)
                y_test_original = scaler.inverse_transform(y_test)
                rmse = mean_squared_error(y_test_original, y_pred, squared=False)
                print(f"Hidden Units: {hidden_units}, Activation: {activation}, Optimizer: {optimizer}, Output Activation: {output_activation}, RMSE: {rmse}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model

print("Best RMSE:", best_rmse)

# 최적의 모델을 사용하여 예측
best_y_pred_normalized = best_model.predict(X_test)
best_y_pred = scaler.inverse_transform(best_y_pred_normalized)
best_rmse = mean_squared_error(y_test_original, best_y_pred, squared=False)
print("Best Model RMSE:", best_rmse)