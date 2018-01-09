import cv2
from PIL import Image
import os
train_img_vid_lst = open("./image_tr_rgb.lst", "w")
prefix = "./image_tr/"
with open("./image_tr.lst") as fp:

    line = fp.readline();
    print(line)
    while line:
        l_array = line.rstrip().split("\t")
        print(l_array)
        img = cv2.imread(l_array[2])
        print(img.shape)
        print(img[0])
        #img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY)
      
        img1 = cv2.cvtColor(cv2.cvtColor(cv2.imread(l_array[2]), cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2GRAY)
        print(img1.shape)
        lala
        img = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        out=Image.fromarray(img,mode="RGB")
        #out.show()
        os.makedirs(os.path.dirname(prefix + l_array[2]), exist_ok=True)
        out.save(prefix + l_array[2])

        train_img_vid_lst.write(l_array[0] + "\t" + l_array[1] + "\t" + prefix + l_array[2] + "\n")

        line = fp.readline();


train_img_vid_lst.close()
