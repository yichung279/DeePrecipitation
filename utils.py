#!/usr/bin/env python3
import numpy as np
from glob import glob
from random import shuffle
from queue import Queue
from PIL import Image
from keras.utils import to_categorical
import cv2
import colorsys

label2pixel = [
    [  0,   0,   0],
    [255, 150,   0],
    [  0,   0, 200],
]

def write_image(filename, img_cls):
    image = np.zeros((img_cls.shape[0], img_cls.shape[1], 3))

    for i in range(img_cls.shape[0]):
        for j in range(img_cls.shape[1]):
            image[i][j][0] = label2pixel[img_cls[i][j]][0]
            image[i][j][1] = label2pixel[img_cls[i][j]][1]
            image[i][j][2] = label2pixel[img_cls[i][j]][2]

    cv2.imwrite(filename, image)

class ImageLoader():

    def __init__(self, cache_size = 10):

        self.cache_size = cache_size

        self.__cache = {}
        self.__file_manager = Queue()
    def __compensate(self, img):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                
                mx =max(img[i][j])
                mn =min(img[i][j])
                s = 0 if mx == 0 else 1- mn/mx
                
                if mx < 150 and s < 0.3:
                #if img[i][j][0] < 200 and img[i][j][0] ==  img[i][j][1] and img[i][j][1] ==  img[i][j][2]:
                    img[i][j] = self.__vote(i, j, img)
        return img
    
    def __vote(self, i, j, img):
        color2rgb = {'white' : (  0,   0,   0),
                     'blue'  : (  0,   0, 200),
                     'green' : (  0, 200,   0),
                     'yellow': (200, 250,   0),
                     'red'   : (200,   0,   0),
                     'purple': (200,   0, 200)}
        color = {'white': 0, 'blue': 0, 'green': 0, 'yellow': 0, 'red': 0, 'purple': 0}

        top = i - 2 if i - 2 >= 0 else 0
        left = j - 2 if j - 2 >= 0 else 0
        bottom = i + 2 if i + 2 < img.shape[0] else img.shape[0] - 1
        right = j + 2 if j + 2 < img.shape[1] else img.shape[1]  - 1

        for m in range(top, bottom + 1):
            for n in range(left, right + 1):
               rgb = [channel/255 for channel in img[m][n]]
               hsv = colorsys.rgb_to_hsv(*rgb)     # hsv = (h, s, v)

               if rgb[0] == rgb[1] and rgb[1] == rgb[2]:     # grayscale
                   color['white'] += 1
               elif hsv[0] > 0.95 or hsv[0] < 0.083:
                   color['red'] += 1
               elif hsv[0] < 0.194:
                   color['yellow'] += 1
               elif hsv[0] < 0.388:
                   color['green'] += 1
               elif hsv[0] < 0.722:
                   color['blue'] += 1
               elif hsv[0] < 0.95:
                   color['purple'] += 1
        
        return color2rgb[max(color, key=color.get)]

    def __get_image_pixel(self, file, area):
        if area == 'A':
            a = 0.5
            b = -1
        elif area == 'B':
            a = 0
            b = 0
        elif area == 'C':
            a = 0
            b = 1
        elif area == 'D':
            a = 0
            b = 2
        elif area == 'E':
            a = 0.5
            b = 3
        elif area == 'F':
            a = 1
            b = 4
 
        if file in self.__cache:
            return self.__cache[file]

        with Image.open(file) as f:
            img_crop = np.array(f.crop((1639 + 72 * a, 1439 - 72 * b, 1711 + 72 * a, 1511 - 72 * b)), dtype = np.uint8)
            img_crop = self.__compensate(img_crop)

            for i in range(img_crop.shape[0]):
                for j in range(img_crop.shape[1]):
                    if img_crop[i][j][0] == img_crop[i][j][1] and img_crop[i][j][0] == img_crop[i][j][2]:
                        img_crop[i][j][0] = 0
                        img_crop[i][j][1] = 0
                        img_crop[i][j][2] = 0

        self.__cache[file] = img_crop
        self.__file_manager.put(file)

        return img_crop

    def read(self, image, area):
        img = self.__get_image_pixel(image, area)

        while self.__file_manager.qsize() > self.cache_size:
            key = self.__file_manager.get()
            del self.__cache[key]

        return img

class DataLoader():

    def __init__(self, file_glob_pattern, batch_size, num_classes = 3):
        self.files = glob(file_glob_pattern)

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.ptr = 1

        shuffle(self.files)

        self.holder = np.load(self.files[0])
        np.random.shuffle(self.holder)

    def load(self):
        chunk = np.load(self.files[self.ptr])
        np.random.shuffle(chunk)

        self.holder = np.concatenate([self.holder, chunk], axis = 0)

        self.ptr += 1

    def __iter__(self):
        return self

    def __next__(self):

        if self.holder.shape[0] < self.batch_size:
            if self.ptr >= len(self.files):
                self.ptr = 0
                shuffle(self.files)

            self.load()

        batch, self.holder = np.split(self.holder, [self.batch_size], axis = 0)
        imgs, y = np.split(batch.astype(float), [9], axis = 3)
        img1, img2, img3 = np.split(imgs, 3, axis = 3)

        x = [0] * imgs.shape[0]
        for idx in range(imgs.shape[0]):
            x[idx] = [img1[idx], img2[idx], img3[idx]]

        return np.array(x), to_categorical(y, num_classes = self.num_classes)

if __name__ == '__main__':

    data_loader = DataLoader(file_glob_pattern = 'feature/train.*.npy', batch_size = 72)
    data = data_loader.__next__()
    print(np.array(data[0]).shape)

