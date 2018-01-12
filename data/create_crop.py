import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pre_crop import *




def create_invert(lst):
    hist_list = []
    if lst == "image_dev.txt":
        test_root = './image_dev/'
    if lst == "image_tr.txt":
        test_root = './image_tr/'


    list_total = open("crop" + lst, "w")

    idx = 0
    idxl = 0
    idxd = 0
    with open(lst) as fp:
        line = fp.readline();
        line = fp.readline();
        while line:
            l_array = line.rstrip().split(",")
            print(l_array)
            im = l_array[0]
            label = str(int(l_array[1]) -1)

            imarr=np.array(Image.open(test_root + im))
            imarr_sum = np.sum(imarr)

            #print(imarr_sum)


            if imarr_sum < 250000000:
                x, y, l, r = pre_crop("image_dev/" + im, 1)
            if imarr_sum > 250000000:
                x, y, l, r = pre_crop("image_dev/" + im, 0)
            im = "crop" + im
            imarr = imarr[x:y, l:r]

            out=Image.fromarray(numpy.array(imarr,dtype=numpy.int8),mode="L")
            #print(out)
            out.save("1/" + im)
            list_total.write(str(idx) + "\t" + label + "\t" + test_root + im +"\n")

            idx = idx + 1
            line = fp.readline();

create_invert("image_dev.txt")
