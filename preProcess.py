from datetime import datetime, timedelta
import sys,time
from glob import glob
from pprint import pprint
from PIL import Image
from queue import Queue
import numpy as np
import re
import time

def is_complete(tstamps):
    complete = True
    delta = timedelta(seconds = 600)
    for i in range(len(tstamps) - 1):
        d1 = datetime.strptime(tstamps[i], "%Y%m%d%H%M")
        d2 = datetime.strptime(tstamps[i + 1], "%Y%m%d%H%M")
        if d1 - d2 != delta:
            complete = False
    return complete

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


if __name__ ==  '__main__':

    files = []
    files += glob('./image_ml/*.jpg')
    files += glob('./image_ml/*.png')
    evenDay = []
    oddDay = []
    
    for f in files:
        if int(f[27])%2 == 0:
            evenDay += [f]
        else:
            oddDay += [f]
    
    evenDay.sort()
    evenDay.reverse()
    oddDay.sort()
    oddDay.reverse()

    oddfile_order = []  # record every 4 continuous files and thier order
    for i in range(len(oddDay) - 3) :
        # four img in a data
        data = [oddDay[i+j] for j in range(4)]
        # check imgs is continuous
        tstamps = [re.search('./image_ml/CV1_3600_([0-9]{12})\.[jpng]{3}', data[j]).group(1)\
                   for j in range(4)]
        if is_complete(tstamps) :
            oddfile_order.append(data)

    print(len(oddfile_order))

    filenames = []
    for i,data in enumerate(oddfile_order) :
        filenames.append(data)

        if (i+1) % 1000 == 0:
            datafile_name = 'radarTrend.train.%d' %((i+1) / 1000)
            make_file(filenames, datafile_name)
            filenames = []

    evenfile_order = []  # record every 5 continuous files and thier order
    for i in range(len(evenDay) - 3) :
        # four img in a data
        data = [evenDay[i+j] for j in range(4)]
        # check imgs is continuous
        tstamps = [re.search('./image_ml/CV1_3600_([0-9]{12})\.[jpng]{3}', data[j]).group(1)\
                   for j in range(4)]
        if is_complete(tstamps) :
            evenfile_order.append(data)

    print(len(evenfile_order))

    filenames = []
    for i,data in enumerate(evenfile_order) :
        filenames.append(data)
        if (i+1) % 1000 == 0 :
            make_file(filenames, 'radarTrend.test.%d' %((i+1) / 1000))
            filenames = []
