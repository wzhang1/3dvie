import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from shutil import copyfile


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def create_invert(lst):
    hist_list = []
    if lst == "image_dev.txt":
        test_root = './image_dev//image_dev/'
        test_root1 = './image_dev1/'
    if lst == "image_tr.txt":
        test_root = './image_tr/image_tr/'
        test_root1 = './image_tr1/'

    os.mkdir(test_root1)
    os.mkdir(test_root1+"1")
    os.mkdir(test_root1+"2")
    os.mkdir(test_root1+"3")



    with open(lst) as fp:
        line = fp.readline();
        line = fp.readline();
        while line:
            l_array = line.rstrip().split(",")
            #print(l_array)
            im = test_root + l_array[0]
            label = l_array[1]
            if label == "1":
                dst = test_root1 + "1/" + l_array[0]
            if label == "2":
                dst = test_root1 + "2/" + l_array[0]
            if label == "3":
                dst = test_root1 + "3/" + l_array[0]
            print(dst)
            copyfile(im, dst)


            #imarr=np.array(Image.open(test_root + im))

            #img = cv2.imread(test_root + im)


           
            line = fp.readline();

#create_invert("image_dev.txt")
create_invert("image_tr.txt")
