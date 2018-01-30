#coding:utf-8

import caffe
import numpy as np
import cv2
import matplotlib.pylab as plt
import sys

if __name__ == '__main__':
    CLASS = int(sys.argv[1])
    net = caffe.Net("../deploy_googlenetCAM_places205.prototxt",
                    "../CAMmodels/imagenet_googlenetCAM_train_iter_2646.caffemodel",
                    caffe.TEST)
    net.forward()

    weights = net.params['CAM_fc'][0].data
    print(weights.shape)
    conv_img = net.blobs['CAM_conv'].data[0]
    print(conv_img.shape)
    weights = np.array(weights,dtype=np.float)
    conv_img = np.array(conv_img,dtype=np.float)
    heat_map = np.zeros([14,14],dtype = np.float)
    for i in range(1024):
        w = weights[CLASS][i]
        heat_map += w*conv_img[i]
    heat_map = cv2.resize(heat_map,(224,224))

    src = cv2.imread('./develop01171.jpg')
    src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    src = cv2.resize(src,(224,224))
    h_min = heat_map.min()
    h_max = heat_map.max()
    heat_map = 256*(heat_map - h_min)/(h_max -h_min)

    print net.blobs['prob'].data[0][CLASS]
    s = plt.imshow(src)
    s = plt.imshow(heat_map,alpha=0.5, interpolation='nearest')
    plt.show()


