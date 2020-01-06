import cv2, keras
from keras.models import Model
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = x_test[0]
y = y_test[0]
x_test = x_test[0].reshape(1, 28, 28, 1).astype('float32')
path = r'D:\python ai\models\hand_write.h5'

model = keras.models.load_model(path)
pre = model.predict(x_test)

print(np.argmax(pre))