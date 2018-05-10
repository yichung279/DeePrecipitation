import numpy as np
import cv2

def postProcess(arr2D, filename):
    img = [[None] * 288] * 288

    for i, row in enumerate(arr2D):
        for j, label in enumerate(row):
            label = np.argmax(label, -1)
            if label == 0:
                pixel = [0, 0, 0]
            elif label == 1:
                pixel = [255, 255, 0]
            elif label == 2:
                pixel = [255, 150, 0]
            elif label == 3:
                pixel = [255, 0, 38]
            elif label == 4:
                pixel = [0, 255, 0]
            elif label == 5:
                pixel = [3, 199, 0]
            elif label == 6:
                pixel = [1, 149, 0]
            elif label == 7:
                pixel = [0, 255, 253]
            elif label == 8:
                pixel = [3, 199, 255]
            elif label == 9:
                pixel = [0, 120, 255]
            elif label == 10:
                pixel = [0, 0, 255]
            elif label == 11:
                pixel = [0, 0, 200]
            elif label == 12:
                pixel = [0, 0, 145]
            elif label == 13:
                pixel = [223, 3, 223]
            elif label == 14:
                pixel = [185, 8, 109]

            img[i][j] = pixel

    img = np.array(img)
    cv2.imwrite(filename, img)

def arr1Dto2D(arr):
    arr2D = [[None] *288] *288
    for i in range(288):
        for j in range(288):
            arr2D[i][j] = arr[288*i + j]
   
    return arr2D

# data.shape = (4, 288*288)
def preProcess(data):
    x=[]
    subX = []

    chanel1 = arr1Dto2D(data[1])
    chanel2 = arr1Dto2D(data[2])
    chanel3 = arr1Dto2D(data[3])
    
    subX.append(chanel1)
    subX.append(chanel2)
    subX.append(chanel3)
    
    x.append(np.transpose(subX))                                                                                
    
    x = np.array(x)
    return x
