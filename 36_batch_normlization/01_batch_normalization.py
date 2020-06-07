import keras
import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization


model = Sequential([
    Dense(units=16, input_shape=(1,5), activation='relu'),
    Dense(units=32, activation='relu'),
    BatchNormalization(axis=1),
    Dense(units=2, activation='softmax')
])

train_labels =  []
train_samples = []

for i in range(50):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)
    
    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)
    
    # The 95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
print('\train_labels:')
print(train_labels)
print('\ntrain_samples:')
print(train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

print('\ntype(scaler):', type(scaler))
print('\nscaled_train_samples:')
print(scaled_train_samples)
