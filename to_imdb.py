import sys
import os
#comment out or change to the python caffe folder
sys.path.insert(0,"/home/w/git/caffe_se/python")

import cv2
import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as mp
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum

def load_label_from_txt(label_file_name):
    labels = {}
    with open(label_file_name) as fp:
        line = fp.readline();
        line = fp.readline();
        while line:
            l_array = line.rstrip().split(",")
            #print(l_array)
            labels[l_array[0]] = int(l_array[1])

            line = fp.readline();
    #print(labels)
    return(labels)


## change to the size of images you want to use
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def write_images_to_lmdb(img_dir, db_name, labels):
    for root, dirs, files in os.walk(img_dir, topdown = False):
        if root != img_dir:
            continue
        #multiply 2 to make the map_size large enough
        map_size = 2*IMAGE_WIDTH * IMAGE_HEIGHT *3*len(files)
        env = lmdb.Environment(db_name, map_size=map_size)
        txn = env.begin(write=True,buffers=True)

        for idx, name in enumerate(files):
            #print(img_dir + name)
            img = cv2.imread(img_dir + name, cv2.IMREAD_COLOR)
            img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
            #print(img.shape)
            #print(img)
            img=img.transpose(2,0,1)
            y = labels[name]
            #print(name)
            #print(y)
            datum = array_to_datum(img,y)
            #print(datum)
            #lala
            str_id = '{:08}'.format(idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())   

            if idx % 1000 == 1:
                print("transforming" + str(idx) + "th image to sb")
    txn.commit()
    env.close()
    print " ".join(["Writing to", db_name, "done!"])

# unzip the tr and dev data to data folder and move the txt labels inside to data folder
tr_labels = load_label_from_txt("./data/image_tr.txt")
write_images_to_lmdb("./data/image_tr/", "image_tr", tr_labels)
dev_labels = load_label_from_txt("./data/image_dev.txt")
write_images_to_lmdb("./data/image_dev/", "image_dev", dev_labels)

