import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
train_labels =  []
train_samples = []
# Example data:
# An experimental drug was tested on individuals from ages 13 to 100.
# The trial had 2100 participants. 
# Half were under 65 years old, half were over 65 years old.
# 95% of patients 65 or older experienced side effects.
# 95% of patients under 65 experienced no side effects.
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
print('\ntrain_labels:', train_labels)
print('\ntrain_samples:', train_samples)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
print('\ntype(scaler):', type(scaler))
print('\nscaled_train_samples:')
print(scaled_train_samples)