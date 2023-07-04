import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import cv2

DATADIRTEST = "E:/Datasets/OclusaisTeste"
CATEGORIESTEST = ["Inferior","Superior"]

NAME = f"Oclusais-Inferiores-vs-Superiores-cnn-64x2-{int(time.time())}"

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))
x_test = []
y_test = []
Xe = []
training_data_Test = []
IMG_SIZE = 50


def create_category_data():
    for categoryTest in CATEGORIESTEST:
        pathTest = os.path.join(DATADIRTEST, categoryTest)
        class_numTest = CATEGORIESTEST.index(categoryTest)
        for imgT in os.listdir(pathTest):
            try:
                img_arrayTest = cv2.imread(os.path.join(pathTest,imgT),cv2.IMREAD_GRAYSCALE)
                new_arrayTest = cv2.resize(img_arrayTest, (IMG_SIZE, IMG_SIZE))
                Xe.append(img_arrayTest)
                training_data_Test.append([new_arrayTest, class_numTest])
            except Exception as e:
                pass

create_category_data()

for features, label in training_data_Test:
    x_test.append(features)
    y_test.append(label)

x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)

#print(X)


X = X/255.0
#x_test = x_test/255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(X, y, batch_size=1, epochs=10, validation_split=0.3, callbacks = [tensorboard])

predictions = model.predict([x_test])
#print(f"{np.argmax(predictions[19])}")
print(len(predictions))


for k in range(len(x_test)):
    fig = plt.figure(figsize=(10, 6))
    img1 = Xe[k]
    fig.add_subplot(1, 2, 1)
    if int(y_test[k]) == 0:
        plt.title("Inferior")
    else:
        plt.title("Superior")
    plt.imshow(img1, cmap='gray')
    img2 = x_test[k]
    fig.add_subplot(1, 2, 2)
    if str(predictions[k]) == "[0.]":
        plt.title("Predicted: Inferior")
    else:
        plt.title("Predicted: Superior")
    plt.imshow(img2, cmap='gray')
    plt.show()
