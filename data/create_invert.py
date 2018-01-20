import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def invert_and_save(test_root, im, img, list_total, list_dark, list_light, is_dark, idx,idxl, idxd, label):
    list_total.write(str(2 * idx) + "\t" + label + "\t" + test_root + im +"\n")

    if is_dark:
        new_im = test_root + "light"+im
        cv2.imwrite(new_im, 255 -img)
        list_dark.write(str(idxd) + "\t" + label + "\t" + new_im +"\n")
        list_total.write(str(2 * idx + 1) + "\t" + label + "\t" + new_im +"\n")
    else:
        new_im = test_root + "dark"+im
        cv2.imwrite(new_im, 255 -img)
        list_light.write(str(idxl) + "\t" + label + "\t" + new_im +"\n")
        list_total.write(str(2 * idx + 1) + "\t" + label + "\t" + new_im +"\n")

def create_invert(lst):
    hist_list = []
    if lst == "image_dev.txt":
        test_root = './image_dev/'
    if lst == "image_tr.txt":
        test_root = './image_tr/'

    list_dark = open("dark" + lst, "w")
    list_light = open("light" + lst, "w")
    list_total = open("total" + lst, "w")

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
            label = l_array[1]

            imarr=np.array(Image.open(test_root + im))
            imarr_sum = np.sum(imarr)
            img = cv2.imread(test_root + im)


            if imarr_sum < 250000000:
                invert_and_save(test_root, im, img, list_total, list_dark, list_light, 1, idx, idxl, idxd, label);
                idxd = idxd + 1
            if imarr_sum > 250000000:
                invert_and_save(test_root, im, img,  list_total, list_dark, list_light, 0, idx,idxl, idxd, label);
                idxl = idxl + 1
            idx = idx + 1
            line = fp.readline();

create_invert("image_dev.txt")
