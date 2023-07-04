import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import numpy as np
import random
import pickle

def change_color_of_cluster(img):
        if len(img.shape) == 2:
                img = np.array(img)
        elif len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_PP_CENTERS
        compactness,labels,centers = cv2.kmeans(np.float32(img.flatten()),2,None,criteria,10,flags)

        labels = np.reshape(labels,img.shape).astype(np.uint8)
        labindex = np.argsort(centers[:,0])
        colorlab = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2BGR)
        # colorlab[labels == labindex[0]] = [0,0,0]
        colorlab[labels == labindex[1]] = [255,255,255]
        return colorlab

view_data = []

#path = 'E:/Datasets/Oclusais/Inferior'

def tratamento(path):
        for img in os.listdir(path):
                print(os.path.join(path,img))
                if "Superior" in os.path.join(path,img):
                        img_array = mpimg.imread(os.path.join(path,img))
                        comp = img_array
                        IMG_SIZE = 10
                        img_array = cv2.resize(img_array, (60, 40))
                        #img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        new_array = []
                        rgb = img_array[:, :, :3]
                        img_array = change_color_of_cluster(img_array)
                        color = [159, 159, 159]  # Original value to be mask
                        black = [0, 0, 0]
                        white = [255, 255, 255]
                        mask = np.all(rgb == [159,159,159], axis=-1)
                        mask += np.all(rgb == [157,157,157], axis=-1)
                        mask += np.all(rgb == [160, 160, 160], axis=-1)
                        mask += np.all(rgb == [158, 158, 158], axis=-1)
                        mask += np.all(rgb == [158, 158, 158], axis=-1)
                        mask += np.all(rgb == [155, 155, 155], axis=-1)
                        mask += np.all(rgb == [62, 62, 62], axis=-1)
                        mask += np.all(rgb == [111, 111, 111], axis=-1)
                        mask += np.all(rgb == [134, 134, 134], axis=-1)
                        mask += np.all(rgb == [39, 39, 39], axis=-1)
                        mask += np.all(rgb == [40, 40, 40], axis=-1)
                        mask += np.all(rgb == [122, 122, 122], axis=-1)
                        mask += np.all(rgb == [142, 142, 142], axis=-1)
                        #img_array[mask] = white
                        #img_array[np.logical_not(mask)] = black
                        #img_array[:, :, 1] = np.zeros([img_array.shape[0], img_array.shape[1]])
                        count = 0
                        for i in range(0,7,1):
                                for k in range (0,10,1):
                                        for j in range(0,3,1):
                                                if (((img_array[i])[k])[j]) != 255 and (((img_array[i])[k])[j]) != 40 and (((img_array[i])[k])[j]) != 0:
                                                        count += 1
                        count = count//3
                        fig = plt.figure(figsize=(10, 6))
                        img1 = comp
                        fig.add_subplot(1, 2, 1)
                        plt.title(img)
                        plt.imshow(img1, cmap='gray')
                        img2 = img_array
                        fig.add_subplot(1, 2, 2)
                        if count == 4:
                                plt.title("Predicted: Lateral")
                        elif count == 6:
                                plt.title("Predicted: Traseira")
                        else:
                                plt.title("Predicted: Superior")
                        plt.imshow(img2, cmap='gray')
                        plt.show()
                        #plt.imshow(img_array)
                        #plt.show()
                        #print(count//3)

path1 = 'E:/TensorFlowPyCharm/Prints/STL'
for p in os.listdir(path1):
        path = os.path.join(path1,p)
        tratamento(path)


#plt.imshow(img_array)

training_data_Test = []
training_data = []