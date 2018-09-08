from datetime import datetime, timedelta
import sys,time
import colorsys
from glob import glob
from PIL import Image
from queue import Queue
import numpy as np
import os
import re
import time

from utils import ImageLoader

def is_complete(tstamps):
    delta = timedelta(seconds = 600)

    if datetime.strptime(tstamps[0], "%Y%m%d%H%M") - datetime.strptime(tstamps[1], "%Y%m%d%H%M") != 2 * delta:
        return False

    for i in range(len(tstamps) - 2):
        if datetime.strptime(tstamps[1 + i], "%Y%m%d%H%M") - datetime.strptime(tstamps[2 + i], "%Y%m%d%H%M") != delta:
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
        match = re.search('CV1_3600_[0-9]{4}([0-9]{2})([0-9]{2})[0-9]{4}\.(jpg|png)', f)
        month = int(match.group(1))
        date = int(match.group(2))

        if date % 2 == 0:
            even_day.append(f)
        else:
            odd_day.append(f)

    even_day.sort()
    even_day.reverse()
    odd_day.sort()
    odd_day.reverse()

    return even_day, odd_day

def get_classification(pixels):

    arr = np.zeros((pixels.shape[0], pixels.shape[1], 1), dtype = np.uint8)

    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            arr[i][j][0] = classify(pixels[i][j])

    return arr


def build_feature(filelist, dest_prefix, area, days = 4):
    features = []

    image_loader = ImageLoader(cache_size = 10)

    idx = 0

    for i in range(len(filelist) - days):
        data = [filelist[i + j] for j in range(days + 1)]
        data.pop(1)
        # check images is continuous
        tstamps = [re.search('CV1_3600_([0-9]{12})\.(jpg|png)', data[j]).group(1) for j in range(len(data))]

        if not is_complete(tstamps):
            continue

        feature = [image_loader.read(image, area) for image in data]

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
    even_day, odd_day, = get_filelist('radar_images')
            
    if not os.path.isdir('no_compensate_feature'):
        os.makedirs('no_compensate_feature')
    for i in 'ABCEDF':
        build_feature(even_day, dest_prefix = 'no_compensate_feature/%s.train' % i, area = i)
        build_feature(odd_day , dest_prefix = 'no_compensate_feature/%s.valid' % i, area = i)
