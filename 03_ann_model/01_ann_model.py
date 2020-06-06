from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(units=32, input_shape=(10,), activation='relu'),
    Dense(units=2, activation='softmax'),
])