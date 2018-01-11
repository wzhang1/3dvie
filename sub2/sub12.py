import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import sys
import numpy as np
import cv2
import json
#from ../common import find_mxnet
import mxnet as mx
import time

START_TIME = time.time()


def ch_dev(arg_params, aux_params, ctx):
    new_args = dict()
    new_auxs = dict()
    for k, v in arg_params.items():
        new_args[k] = v.as_in_context(ctx)
    for k, v in aux_params.items():
        new_auxs[k] = v.as_in_context(ctx)
    return new_args, new_auxs
'''
def resize_short(src, size, interp=2):
    """Resizes shorter edge to size.
    """
    h, w, _ = src.shape
    if h > w:
        new_h, new_w = size * h / w, size
    else:
        new_h, new_w = size, size * w / h
    return imresize(src, new_w, new_h, interp=interp)

def center_crop(src, size, interp=2):
    """Crops the image `src` to the given `size` by trimming on all four
    """

    h, w, _ = src.shape
    new_w, new_h = scale_down((w, h), size)

    x0 = int((w - new_w) / 2)
    y0 = int((h - new_h) / 2)

    out = fixed_crop(src, x0, y0, new_w, new_h, size, interp)
    return out, (x0, y0, new_w, new_h)
'''

def predict(prefix, epoch, gpu_id, test_lst_file, img_sz, crop_sz, batch_sz=64):
    #prefix = '../model_level3/3dvie-resnet-18bn24'
    #epoch = int(sys.argv[1]) #check point step
    #gpu_id = int(sys.argv[2]) #GPU ID for infer
    ctx = mx.gpu(gpu_id)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)

    #test_folder = '../data/image_dev/'
    test_folder = '../data/'
    #imgs = [aa['image_id'] for aa in ann_data]
    imgs = []
    with open(test_lst_file) as fp:
        line = fp.readline();
        while line:
            #l_array = line.rstrip().split(",")
            l_array = line.rstrip().split("\t")
            print(l_array)
            
            imgs.append(l_array[2])
            line = fp.readline();

    classes = [0]*len(imgs)
    cnt = 0

    #img_sz = int(sys.argv[3])
    #crop_sz = int(sys.argv[3])

    preds = []
    im_idxs = []


    im_probs = []
    result=[]

    #szx = 337
    #szy = 600
    szx = 600
    szy = 337
    #batch_sz = 64
    #input_blob = np.zeros((batch_sz,3,szx,crop_szy))
    input_blob = np.zeros((batch_sz,3, 600, 337))
    idx = 0
    num_batches = int(len(imgs) / batch_sz)


    for batch_head in range(0, batch_sz*num_batches, batch_sz):
    #for batch_head in range(0, batch_sz*2, batch_sz):
        #print batch_head
        for index in range(batch_head, batch_head+batch_sz):
	    img_name = imgs[index]
            label = str(classes[index])
            im_id = img_name
            im_idxs.append(im_id)

            cnt += 1
            img_full_name = test_folder  + img_name
            #print(img_full_name)
            img = cv2.cvtColor(cv2.imread(img_full_name), cv2.COLOR_BGR2RGB)
            img= cv2.resize(img, (337, 600))
            img = np.float32(img)
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)  # change to r,g,b order
            #print(img.shape)
            #print(input_blob.shape)
            input_blob[idx,:,:,:] = img
            idx += 1
	    #print(idx)

        idx = 0


        arg_params["data"] = mx.nd.array(input_blob, ctx)
        arg_params["softmax_label"] = mx.nd.empty((batch_sz,), ctx)
        exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
        exe.forward(is_train=False)
        net_out = exe.outputs[0].asnumpy()

        for bz in range(batch_sz):
	    probs = net_out[bz,:]
    	    score = np.squeeze(probs)
            print(probs)
            im_probs.append(probs)

            sort_index = np.argsort(score)[::-1]
            top_k = sort_index[0:2]
            preds.append(top_k.astype(np.int))
	    #print(preds[-1], batch_head+bz)



    for index in range(batch_sz*num_batches, len(imgs)):
        img_name = imgs[index]
        label = str(classes[index])
        im_id = img_name
        im_idxs.append(im_id)
        cnt += 1
        img_full_name = test_folder + img_name
        img = cv2.cvtColor(cv2.imread(img_full_name), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (337, 600))
        img = np.float32(img)
        img = mx.nd.array(img)
                    
        #img = mx.img.resize_short(img, 337 * 6 /5)
        #img = mx.image.center_crop(img, (600, 337))

        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # change to r,g,b order

        img = img[np.newaxis, :]
        arg_params["data"] = mx.nd.array(img, ctx)
        #arg_params["data"] = mx.nd.array(input_blob, ctx)
        arg_params["softmax_label"] = mx.nd.empty((1,), ctx)
        exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
        exe.forward(is_train=False)
        probs = exe.outputs[0].asnumpy()
        print(probs)
        im_probs.append(probs)
        score = np.squeeze(probs.mean(axis=0))

        sort_index = np.argsort(score)[::-1]
        top_k = sort_index[0:2]
        #print(top_k)

        preds.append(top_k.astype(np.int))
	#print(preds[-1], im_idxs[-1])
	#print(top_k.astype(np.int), int(im_id))
	#print(preds[index], im_idxs[index])

    im_idxs = np.hstack(im_idxs)
    preds = np.vstack(preds)
    im_probs = np.vstack(im_probs)

    return im_idxs, preds, im_probs

#im_idxs12, preds12, im_probs12 = predict("../model_12/3dvie-resnet-152", 25, 1, "12new440x440-ep15.csv", 440, 550, 4)
#im_idxs12, preds12, im_probs12 = predict("../model_level3/3dvie-resnet-18_12", 10, 1, "../data/image_dev12.csv", 440, 440, 4)
im_idxs12, preds12, im_probs12 = predict("../model_12/3dvie-resnet-152", 3, 0, "../data/image_dev12.lst", 337, 600, 64)
print(im_idxs12)
print(len(preds12))
print(len(im_probs12))
print(len(im_idxs12))
print(len(preds12))
print(len(im_probs12))


result=[]

with open("partial"+"12.csv" , 'w') as opfile:
    opfile.write('id,predicted\n')
    for ii in range(len(im_idxs12)):
        opfile.write(str(im_idxs12[ii]) + ',' + ' '.join(str(x) for x in preds12[ii,:])+'\n')
        temp_dict = {}

        temp_dict['label_id'] = preds12[ii,:].tolist()
        temp_dict['image_id'] = str(im_idxs12[ii])
        result.append(temp_dict)

