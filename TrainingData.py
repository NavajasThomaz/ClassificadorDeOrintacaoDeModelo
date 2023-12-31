import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle


DATADIR = "E:/Datasets/Oclusais"
CATEGORIES = ["Inferior","Superior"]

DATADIRTEST = "E:/Datasets/OclusaisTeste"
CATEGORIESTEST = ["Inferior","Superior"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
view_data = []
for categoryTest in CATEGORIESTEST:
    pathtest = os.path.join(DATADIRTEST, categoryTest)
    for imgTest in os.listdir(pathtest):
        try:
            img_arrayTest = cv2.imread(os.path.join(pathtest, imgTest), cv2.IMREAD_GRAYSCALE)
            view_data.append(img_arrayTest)
        except Exception as e:
            pass

pickle_out = open("Xe.pickle","wb")
pickle.dump(view_data, pickle_out)
pickle_out.close()

IMG_SIZE = 50
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
new_arrayTest = cv2.resize(img_arrayTest, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap = 'gray')
plt.show()

training_data_Test = []
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()



print(len(training_data))
random.shuffle(training_data)
#random.shuffle(training_data_Test)
#for sample in training_data:
    #print(f"{sample[1]}, ")

X = []
y = []
Xt = []
yt = []

for features, label in training_data:
    X.append(features)
    y.append(label)

for features, label in training_data_Test:
    Xt.append(features)
    yt.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

Xt = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
yt = np.array(y)
print(len(Xt))

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open("Xt.pickle","wb")
pickle.dump(Xt, pickle_out)
pickle_out.close()

pickle_out = open("yt.pickle","wb")
pickle.dump(yt, pickle_out)
pickle_out.close()

#pickle_in = open("X.pickle","rb")
#X = pickle.load(pickle_in)
#print(X)