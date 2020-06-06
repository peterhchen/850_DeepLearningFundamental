from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(units=5, input_shape=(3,), activation='relu')
])
model = Sequential()
model.add(Dense(units=5, input_shape=(3,)))
model.add(Activation('relu'))
