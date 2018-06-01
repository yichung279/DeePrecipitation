from datetime import datetime, timedelta
import sys,time
import colorsys
from glob import glob
from pprint import pprint
from PIL import Image
from queue import Queue
import numpy as np
import os
import re
import time

from utils import ImageLoader

def is_complete(tstamps):
    delta = timedelta(seconds = 600)

    for i in range(len(tstamps) - 1):
        if datetime.strptime(tstamps[i], "%Y%m%d%H%M") - datetime.strptime(tstamps[i + 1], "%Y%m%d%H%M") != delta:
            return False

    return True

#!TODO: classification logic
def classify(pixel) :
    if np.array_equal([0, 0, 0], pixel):
        return 0
    
    rgb = [channel/255 for channel in pixel]
    hsv = colorsys.rgb_to_hsv(*rgb)
    hue = hsv[0]
    
    if 0.5 <= hue and hue <= 0.7:
        return 1
    else:
        return 2

def get_filelist(directory):
    files = glob('%s/*.jpg' % directory)
    files.extend(glob('%s/*.png' % directory))

    even_day = []
    odd_day = []

    for f in files:
        match = re.search('CV1_3600_[0-9]{6}([0-9]{2})[0-9]{4}\.(jpg|png)', f)
        date = int(match.group(1))

        if date % 2 == 0:
            even_day.append(f)
        else:
            odd_day.append(f)

    even_day.sort()
    even_day.reverse()
    odd_day.sort()
    odd_day.reverse()

    return even_day, odd_day

def get_image_pixel(file):
    with Image.open(file) as f:
        img_crop = np.array(f.crop((1639, 1439, 1711, 1511)), dtype = np.uint8)

        for i in range(img_crop.shape[0]):
            for j in range(img_crop.shape[1]):
                if img_crop[i][j][0] == img_crop[i][j][1] and img_crop[i][j][0] == img_crop[i][j][2]:
                    img_crop[i][j][0] = 0
                    img_crop[i][j][1] = 0
                    img_crop[i][j][2] = 0

    return img_crop

def get_classification(pixels):

    arr = np.zeros((pixels.shape[0], pixels.shape[1], 1), dtype = np.uint8)

    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            arr[i][j][0] = classify(pixels[i][j])

    return arr


def build_feature(filelist, dest_prefix, days = 3):
    features = []

    image_loader = ImageLoader(cache_size = 10)

    idx = 0

    for i in range(len(filelist) - days):
        data = [filelist[i + j] for j in range(days + 1)]

        # check images is continuous
        tstamps = [re.search('CV1_3600_([0-9]{12})\.(jpg|png)', data[j]).group(1) for j in range(days + 1)]

        if not is_complete(tstamps):
            continue

        # feature = [get_image_pixel(image) for image in data]
        feature = [image_loader.read(image) for image in data]

        label = get_classification(feature[0])
        feature = np.concatenate(feature[1:] + [label], axis = 2)

        features.append(feature)

        if len(features) % 1000 == 0:
            print(idx)
            np.save('%s.%d.npy' % (dest_prefix, idx), np.stack(features, axis = 0))

            idx += 1

            features = []

    np.save('%s.%d.npy' % (dest_prefix, idx), np.stack(features, axis = 0))

if __name__ ==  '__main__':
    '''    
    test classify()
    pixels = [
        [  0,   0,   0],        # 0
        [  0,  51, 245],        # 1
        [  0,  10, 100],        # 1
        [  0,  20,   0],        # 2
        [ 30, 200,   0],        # 2
        [200,   0, 100]         # 2
    ] 

    for pixel in pixels:
        print(classify(np.array(pixel)))
    '''
    even_day, odd_day = get_filelist('image_ml')

    if not os.path.isdir('feature'):
        os.makedirs('feature')

    build_feature(even_day, dest_prefix = 'feature/train')
    build_feature(odd_day, dest_prefix = 'feature/valid')
