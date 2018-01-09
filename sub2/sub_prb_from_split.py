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

def resize(img, img_sz, crop_sz):

    rows, cols = img.shape[:2]
    if cols < rows:
        resize_width = img_sz
        resize_height = resize_width * rows / cols;
    else:
        resize_height = img_sz
        resize_width = resize_height * cols / rows;
    print(resize_width, resize_height)
    img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
    print(img.shape)
    #batch = oversample(img, (crop_sz,crop_sz))

    h, w, _ = img.shape

    x0 = int((w - crop_sz) / 2)
    y0 = int((h - crop_sz) / 2)
    print(x0, y0)
    print(y0,y0+crop_sz, x0,x0+crop_sz)
    y0 = max(0, y0)
    img = img[y0:y0+crop_sz, x0:x0+crop_sz]
    print(img.shape)
    return img

def predict(prefix, epoch, gpu_id, test_lst_file, img_sz, crop_sz, batch_sz=64):
    #prefix = '../model_level3/3dvie-resnet-18bn24'
    #epoch = int(sys.argv[1]) #check point step
    #gpu_id = int(sys.argv[2]) #GPU ID for infer
    ctx = mx.gpu(gpu_id)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)

    test_folder = '../data/image_dev/'

    #imgs = [aa['image_id'] for aa in ann_data]
    imgs = []
    with open(test_lst_file) as fp:
        line = fp.readline();
        while line:
            l_array = line.rstrip().split(",")
            imgs.append(l_array[0])
            line = fp.readline();

    classes = [0]*len(imgs)
    cnt = 0

    #img_sz = int(sys.argv[3])
    #crop_sz = int(sys.argv[3])

    preds = []
    im_idxs = []


    im_probs = []
    result=[]


    #batch_sz = 64
    input_blob = np.zeros((batch_sz,3,crop_sz,crop_sz))
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
            img = np.float32(img)
            #(1080, 1920,3)
            #print(img.shape)
            img = cv2.resize(img, (crop_sz,crop_sz), interpolation=cv2.INTER_CUBIC)
            #img = resize(img, img_sz, crop_sz)
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)  # change to r,g,b order
            
            input_blob[idx,:,:,:] = img
            idx += 1
	    #print(idx)

        idx = 0


        arg_params["data"] = mx.nd.array(input_blob, ctx)
        arg_params["softmax_label"] = mx.nd.empty((batch_sz,), ctx)
        exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
        exe.forward(is_train=False)
        net_out = exe.outputs[0].asnumpy()

        input_blob = np.zeros((batch_sz,3,crop_sz,crop_sz))

        for bz in range(batch_sz):
	    probs = net_out[bz,:]
    	    score = np.squeeze(probs)
            #print(probs)
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
        img = np.float32(img)
                    
        #img = cv2.resize(img, (crop_sz,crop_sz), interpolation=cv2.INTER_CUBIC)
        img = resize(img, img_sz, crop_sz)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # change to r,g,b order

        img = img[np.newaxis, :]
        arg_params["data"] = mx.nd.array(img, ctx)
        #arg_params["data"] = mx.nd.array(input_blob, ctx)
        arg_params["softmax_label"] = mx.nd.empty((1,), ctx)
        exe = sym.bind(ctx, arg_params ,args_grad=None, grad_req="null", aux_states=aux_params)
        exe.forward(is_train=False)
        probs = exe.outputs[0].asnumpy()
        #print(probs)
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

im_idxs23, preds23, im_probs23 = predict("../model_level3/3dvie-resnet-152_23", 10, 0, "23new440x440-ep15.csv", 440, 550, 4)
#print(preds23)
def minus_one(x):
    return(x + 1)
minus_one = np.vectorize(minus_one)
preds23 = minus_one(preds23)
print(preds23)
im_idxs13, preds13, im_probs13 = predict("../model_level3/3dvie-resnet-18_13", 2, 0, "13new440x440-ep15.csv", 300, 300, 4)
def convert13(x):
    if x == 1:
        return(x + 1)
    return(x)
convert13 = np.vectorize(convert13)
preds13 = convert13(preds13)
print(preds13)
im_idxs12, preds12, im_probs12 = predict("../model_12/3dvie-resnet-152", 25, 1, "12new440x440-ep15.csv", 440, 550, 4)
print(im_idxs12)
print(len(preds12))
print(len(im_probs12))
print(len(im_idxs12))
print(len(preds12))
print(len(im_probs12))



result=[]
im_idxs = np.concatenate([im_idxs12, im_idxs23,im_idxs13])
preds = np.concatenate([preds12, preds23, preds13])
im_probs = np.concatenate([im_probs12, im_probs23, im_probs13])
print(len(im_idxs))
print(len(preds))
print(len(im_probs))

res_prefix="Base3_classes300x300-ep15.csv"
with open(res_prefix+".csv" , 'w') as opfile:
    opfile.write('id,predicted\n')
    for ii in range(len(im_idxs)):
        opfile.write(str(im_idxs[ii]) + ',' + ' '.join(str(x) for x in preds[ii,:])+'\n')
        temp_dict = {}

        temp_dict['label_id'] = preds[ii,:].tolist()
        temp_dict['image_id'] = str(im_idxs[ii])
        result.append(temp_dict)

with open(res_prefix+"12.csv" , 'w') as opfile:
    opfile.write('id,predicted\n')
    for ii in range(len(im_idxs12)):
        opfile.write(str(im_idxs12[ii]) + ',' + ' '.join(str(x) for x in preds12[ii,:])+'\n')
        temp_dict = {}

        temp_dict['label_id'] = preds12[ii,:].tolist()
        temp_dict['image_id'] = str(im_idxs12[ii])
        result.append(temp_dict)
with open(res_prefix+"23.csv" , 'w') as opfile:
    opfile.write('id,predicted\n')
    for ii in range(len(im_idxs23)):
        opfile.write(str(im_idxs23[ii]) + ',' + ' '.join(str(x) for x in preds23[ii,:])+'\n')
        temp_dict = {}

        temp_dict['label_id'] = preds23[ii,:].tolist()
        temp_dict['image_id'] = str(im_idxs23[ii])
        result.append(temp_dict)

with open(res_prefix+"prob.csv", 'w') as opfile:
    #opfile.write('id,predicted\n')
    opfile.write('id,')
    for jj in range(3):
        opfile.write(str(jj) +',')
    opfile.write('\n')
    for ii in range(len(im_idxs)):
        opfile.write(str(im_idxs[ii]) + ',' + ' '.join(str(x) +"," for x in im_probs[ii,:])+'\n')



    print('testing time: %f s' % (time.time() - START_TIME))

