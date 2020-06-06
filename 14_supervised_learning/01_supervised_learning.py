# Supervised Learning
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
import numpy as np

model = Sequential([
    Dense(units=16, input_shape=(2,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), \
    loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# weight, height
train_samples = \
    np.array([[150, 67], [130, 60], [200, 65], [125, 52], [230, 72], [181, 70]])

# 0: male
# 1: female
train_labels = [1, 1, 0, 1, 0, 0]

model.fit(x=train_samples, y=train_labels, batch_size=3, \
    epochs=10, shuffle=True, verbose=2)

print('\ntrain_samples:')
print(train_samples)
print('\ntrain_labels:', train_labels)