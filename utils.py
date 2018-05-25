#!/usr/bin/env python3
import numpy as np
from glob import glob
from random import shuffle
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

class DataLoader():

    def __init__(self, feature_glob_pattern, label_glob_pattern, batch_size):
        feature_files = glob(feature_glob_pattern)
        label_files = glob(label_glob_pattern)

        feature_files.sort()
        label_files.sort()

        self.batch_size = batch_size
        self.file_pairs = []
        self.ptr = 1

        for x, y in zip(feature_files, label_files):
            self.file_pairs.append((x, y))

        shuffle(self.file_pairs)

        self.x_hold = np.load(self.file_pairs[0][0])
        self.y_hold = np.load(self.file_pairs[0][1])

        np.random.shuffle(self.x_hold)
        np.random.shuffle(self.y_hold)

    def load(self):
        x_load = np.load(self.file_pairs[self.ptr][0])
        np.random.shuffle(x_load)

        y_load = np.load(self.file_pairs[self.ptr][1])
        np.random.shuffle(y_load)

        self.x_hold = np.concatenate([self.x_hold, x_load], axis = 0)
        self.y_hold = np.concatenate([self.y_hold, y_load], axis = 0)

        self.ptr += 1

    def __iter__(self):
        return self

    def __next__(self):

        if self.x_hold.shape[0] < self.batch_size:
            if self.ptr < len(self.file_pairs):
                self.load()
            else:
                raise StopIteration

        x_batch, self.x_hold = np.split(self.x_hold, [self.batch_size], axis = 0)
        y_batch, self.y_hold = np.split(self.y_hold, [self.batch_size], axis = 0)

        return x_batch, y_batch

if __name__ == '__main__':

    data_loader = DataLoader(feature_glob_pattern = 'feature/train.feature.*npy',
                             label_glob_pattern = 'feature/train.label.*npy',
                             batch_size = 20)

    for x, y in data_loader:
        print(x.shape)
