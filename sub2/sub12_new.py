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


def load_batch(imgs, batch_head, batch_sz, size_x, size_y, cnt, test_folder):
    # load batch to input_blob
    idx = 0
    input_blob = np.zeros((batch_sz,3, size_y, size_x))
    im_idxs = []

    for index in range(batch_head, batch_head+batch_sz):
	img_name = imgs[index]
        im_id = img_name
        im_idxs.append(im_id)

        cnt += 1
        img_full_name = test_folder  + img_name

        img = cv2.cvtColor(cv2.imread(img_full_name), cv2.COLOR_BGR2RGB)
        img= cv2.resize(img, (size_x, size_y))
        img = np.float32(img)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)

        input_blob[idx,:,:,:] = img
        idx += 1

    return(input_blob, im_idxs, cnt)


def predict(prefix, epoch, gpu_id, test_lst_file, size_x, size_y, crop_sz, batch_sz=64):
    #prefix = '../model_level3/3dvie-resnet-18bn24'
    #epoch = int(sys.argv[1]) #check point step
    #gpu_id = int(sys.argv[2]) #GPU ID for infer
    ctx = mx.gpu(gpu_id)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)

    #test_folder = '../data/image_dev/'
    test_folder = '../data/'

    # load images list
    imgs = []
    with open(test_lst_file) as fp:
        line = fp.readline();
        while line:
            #l_array = line.rstrip().split(",")
            l_array = line.rstrip().split("\t")
            #print(l_array)
            
            imgs.append(l_array[2])
            line = fp.readline();

    classes = [0]*len(imgs)
    cnt = 0

    preds = []
    im_idxs = []


    im_probs = []
    result=[]

    szx = 337
    szy = 600
    szx = 600
    szy = 337
    #batch_sz = 64
    #input_blob = np.zeros((batch_sz,3,szx,crop_szy))
    input_blob = np.zeros((batch_sz,3, 600, 337))
    batch_head = 0
    cnt = 0
    num_batches = int(len(imgs) / batch_sz)
    total_cases = len(imgs)

    #for batch_head in range(0, batch_sz*num_batches, batch_sz):
    #for batch_head in range(0, batch_sz*2, batch_sz):
        #print batch_head
    while total_cases > 0:
        print(total_cases)
        cur_batch_sz = min(total_cases, batch_sz)

        input_blob, im_idx_batch, cnt= load_batch(imgs, batch_head, cur_batch_sz, size_x, size_y, cnt, test_folder)

        arg_params["data"] = mx.nd.array(input_blob, ctx)
        arg_params["softmax_label"] = mx.nd.empty((cur_batch_sz,), ctx)
        exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
        exe.forward(is_train=False)
        net_out = exe.outputs[0].asnumpy()

        for bz in range(cur_batch_sz):
	    probs = net_out[bz,:]
    	    score = np.squeeze(probs)
            #print(probs)
            im_probs.append(probs)

            sort_index = np.argsort(score)[::-1]
            top_k = sort_index[0:2]
            preds.append(top_k.astype(np.int))
	    #print(preds[-1], batch_head+bz)

        im_idxs = im_idxs + im_idx_batch
        total_cases = total_cases - cur_batch_sz
        batch_head = batch_head + cur_batch_sz


    im_idxs = np.hstack(im_idxs)
    preds = np.vstack(preds)
    im_probs = np.vstack(im_probs)

    return im_idxs, preds, im_probs

im_idxs12, preds12, im_probs12 = predict("../model_12/3dvie-resnet-152", 3, 0, "../data/image_dev12.lst", 337, 600, 128)
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



