import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
dataset = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = dataset.load_data()
X_train= X_train/255.0
X_test= X_test/255.0

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

model = Sequential()
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['acc'])

model.fit(X_train, y_train, epochs=5, batch_size=12, validation_split=0.1)
model.save('digit_trained.keras')
run = False
ix,iy = -1,-1
follow = 25
img = np.zeros((512,512,1))

# 