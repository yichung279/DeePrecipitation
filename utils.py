#!/usr/bin/env python3
import numpy as np
from queue import Queue
from PIL import Image

class ImageLoader():

    def __init__(self, cache_size = 10):

        self.cache_size = cache_size

        self.__cache = {}
        self.__file_manager = Queue()

    def __get_image_pixel(self, file):
        if file in self.__cache:
            return self.__cache[file]

        with Image.open(file) as f:
            img_crop = np.array(f.crop((1639, 1439, 1711, 1511)), dtype = np.uint8)

            for i in range(img_crop.shape[0]):
                for j in range(img_crop.shape[1]):
                    if img_crop[i][j][0] == img_crop[i][j][1] and img_crop[i][j][0] == img_crop[i][j][2]:
                        img_crop[i][j][0] = 0
                        img_crop[i][j][1] = 0
                        img_crop[i][j][2] = 0

        self.__cache[file] = img_crop
        self.__file_manager.put(file)

        return img_crop

    def read(self, image):
        img = self.__get_image_pixel(image)

        while self.__file_manager.qsize() > self.cache_size:
            key = self.__file_manager.get()
            del self.__cache[key]

        return img
