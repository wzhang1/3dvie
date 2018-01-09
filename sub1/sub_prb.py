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

def oversample(images, crop_dims):

    im_shape = np.array(images.shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # print crops_ix

    # Extract crops
    crops = np.empty((10, crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    # for im in images:
    im = images
    # print im.shape
    for crop in crops_ix:
        # print crop
        crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
        # cv2.imshow('crop', im[crop[0]:crop[2], crop[1]:crop[3], :])
        # cv2.waitKey()
        ix += 1
    crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]
    # cv2.imshow('crop', crops[0,:,:,:])
    # cv2.waitKey()
    return crops




prefix = '../model_level3-35degree/3dvie-squeezenet_v1.1'
#prefix = '../model_level3/3dvie-resnet-18bn24'

epoch = int(sys.argv[1]) #check point step
gpu_id = int(sys.argv[2]) #GPU ID for infer
ctx = mx.gpu(gpu_id)
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)

test_folder = '../data/image_dev/'
ann_file = '../data/dev_annotations.json'


print('Loading annotations from: ' + os.path.basename(ann_file))
with open(ann_file) as data_file:
    ann_data = json.load(data_file)

imgs = [aa['image_id'] for aa in ann_data]
im_ids = [aa['image_id'] for aa in ann_data]

classes = [0]*len(imgs)





top1_acc = 0
top5_acc = 0
cnt = 0

img_sz = int(sys.argv[3])
crop_sz = int(sys.argv[3])

preds = []
im_idxs = []


im_probs = []
result=[]


batch_sz = 64
input_blob = np.zeros((batch_sz,3,crop_sz,crop_sz))
idx = 0
num_batches = int(len(imgs) / batch_sz)


for batch_head in range(0, batch_sz*num_batches, batch_sz):
#for batch_head in range(0, batch_sz*2, batch_sz):
    #print batch_head
    for index in range(batch_head, batch_head+batch_sz):
	img_name = imgs[index]
        label = str(classes[index])
        im_id = str(im_ids[index])
        im_idxs.append(im_id)

        cnt += 1
        img_full_name = test_folder  + img_name
        #print(img_full_name)
        img = cv2.cvtColor(cv2.imread(img_full_name), cv2.COLOR_BGR2RGB)
        img = np.float32(img)

        rows, cols = img.shape[:2]
        if cols < rows:
            resize_width = img_sz
            resize_height = resize_width * rows / cols;
        else:
            resize_height = img_sz
            resize_width = resize_height * cols / rows;

        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)

	h, w, _ = img.shape

        x0 = int((w - crop_sz) / 2)
        y0 = int((h - crop_sz) / 2)
        img = img[y0:y0+crop_sz, x0:x0+crop_sz]

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
        top_k = sort_index[0:3]
        preds.append(top_k.astype(np.int))
	#print(preds[-1], batch_head+bz)



for index in range(batch_sz*num_batches, len(imgs)):
	img_name = imgs[index]
        label = str(classes[index])
        im_id = str(im_ids[index])
        im_idxs.append(im_id)
        cnt += 1
        img_full_name = test_folder + img_name
        img = cv2.cvtColor(cv2.imread(img_full_name), cv2.COLOR_BGR2RGB)
        img = np.float32(img)

        rows, cols = img.shape[:2]
        if cols < rows:
            resize_width = img_sz
            resize_height = resize_width * rows / cols;
        else:
            resize_height = img_sz
            resize_width = resize_height * cols / rows;

        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)

        #batch = oversample(img, (crop_sz,crop_sz))

        h, w, _ = img.shape

        x0 = int((w - crop_sz) / 2)
        y0 = int((h - crop_sz) / 2)
        img = img[y0:y0+crop_sz, x0:x0+crop_sz]

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
        top_k = sort_index[0:3]
	print(top_k)

        preds.append(top_k.astype(np.int))
	#print(preds[-1], im_idxs[-1])
	#print(top_k.astype(np.int), int(im_id))
	#print(preds[index], im_idxs[index])

im_idxs = np.hstack(im_idxs)
preds = np.vstack(preds)
im_probs = np.vstack(im_probs)

result=[]
print(len(im_idxs))
print(len(preds))
print(len(im_probs))

prefix="new" + str(int(sys.argv[3])) + "x"+str(int(sys.argv[3]))+"-ep"+str(int(sys.argv[1]))
with open(prefix+".csv" , 'w') as opfile:
    opfile.write('id,predicted\n')
    for ii in range(len(im_idxs)):
        opfile.write(str(im_idxs[ii]) + ',' + ' '.join(str(x) for x in preds[ii,:])+'\n')
        temp_dict = {}

        temp_dict['label_id'] = preds[ii,:].tolist()
        temp_dict['image_id'] = str(im_idxs[ii])
        result.append(temp_dict)


with open(prefix+"prob.csv", 'w') as opfile:
    #opfile.write('id,predicted\n')
    opfile.write('id,')
    for jj in range(3):
        opfile.write(str(jj) +',')
    opfile.write('\n')
    for ii in range(len(im_idxs)):
        opfile.write(str(im_idxs[ii]) + ',' + ' '.join(str(x) +"," for x in im_probs[ii,:])+'\n')


with open(prefix +".json", 'w') as f:
    json.dump(result, f)
    print('write result json, num is %d' % len(result))


    print('testing time: %f s' % (time.time() - START_TIME))
