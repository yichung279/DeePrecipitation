from datetime import datetime, timedelta
import sys,time
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

# Deprecate
def classify(pixel) :
    #grayscale
    if pixel[0] ==pixel[1] and pixel[1] == pixel[2]:
        return 0
        # classify(1~14) by color (blue->green->yellow->red->purple)
    #blue
    elif pixel[0] <= 50 and pixel[2] >= 200:
        if  pixel[1] >= 200:
            return 1
        elif  pixel[1] >= 100:
            return 2
        else:
            return 3
    #green
    elif pixel[0] <= 50 and pixel[2] <= 50 :
        if  pixel[1] >= 225:
            return 4 
        elif  pixel[1] >= 175:
            return 5
        else:
            return 6
    #orange
    elif pixel[0] >= 225 and pixel[2] <= 50 :
        if  pixel[1] >= 225:
            return 7
        elif  pixel[1] >= 175:
            return 8
        elif  pixel[1] >= 100:
            return 9
        else:
            return 10
    #dark red
    elif pixel[1] <= 50 and pixel[2] <= 50:
        if pixel[0] >= 175:
            return 11
        else :
            return 12
    #purple
    elif pixel[1] <= 50 and pixel[2] >= 200:
        if pixel[0] >= 175:
            return 13
        else :
            return 14
    else:
        return 0

# filenames(1000, 4)
# Deprecate
def make_file(filenames, datafile_name) :

    print(datafile_name + ':')    

    labeldata = [] 
    inputdata = []
    
    for p, filename in enumerate(filenames) :
        if p % 20 == 0:
            sys.stdout.write('>')
            sys.stdout.flush()

        train_data = [[[None, None, None]]*144]*144
        for x in range(1, 4) :
            # !Todo: config position of filename
            with Image.open(filename[x]) as img :
                pixels = img.crop((1639, 1439, 1711, 1511)).load()
                for i in range(72) :
                    for j in range(72) :
                        color = classify(pixels[i,j])
                        train_data[i][j][x-1] = color
        inputdata.append(train_data)

        label = [[[0]*15 for i in range(144)] for j in range(144)]
        with Image.open(filename[0]) as img:
            pixels = img.crop((1639, 1439, 1711, 1511)).load()
            for i in range(72) :
                for j in range(72) :
                    color = classify(pixels[i,j])
                    label[i][j][color] = 1
        labeldata.append(label)

    labeldata = np.array(labeldata, dtype=np.uint8)
    np.save('data/'+datafile_name+'.label.npy', labeldata)

    inputdata = np.array(inputdata, dtype=np.uint8)
    np.save('data/'+datafile_name+'.input.npy', inputdata)

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

    arr = np.zeros((pixels.shape[0], pixels.shape[1]), dtype = np.uint8)

    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            #!TODO: More classification
            if pixels[i][j][0] >= 175 and pixels[i][j][1] <= 50 and pixels[i][j][2] <= 50:
                arr[i][j] = 1

    return arr


def build_feature(filelist, dest_prefix, days = 3):
    features = []
    labels = []

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
        feature = np.concatenate(feature[1:], axis = 2)

        features.append(feature)
        labels.append(label)

        print(len(features))

        if len(features) % 1000 == 0:
            np.save('%s.feature.%d.npy' % (dest_prefix, idx), np.stack(features, axis = 0))
            np.save('%s.label.%d.npy' % (dest_prefix, idx), np.stack(labels, axis = 0))

            idx += 1

            features = []
            labels = []

if __name__ ==  '__main__':

    even_day, odd_day = get_filelist('image_ml')

    if not os.path.isdir('feature'):
        os.makedirs('feature')

    build_feature(even_day, dest_prefix = 'feature/train')
    build_feature(odd_day, dest_prefix = 'feature/valid')
