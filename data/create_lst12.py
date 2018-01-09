# iNatularist image loader


from PIL import Image
import os
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def default_loader(path):
    return Image.open(path).convert('RGB')

def gen_list(prefix):
    ann_file = '%s.txt'%prefix
    train_out = '%s12.lst'%prefix
    # load annotations

    df = pd.read_csv(ann_file)

    cols = df.columns.tolist()
    print(cols)

    df= df[['labels', 'name']]
    df['name'] = prefix + '/' +df['name']
    df = (df.loc[df['labels'] != 3])
    df['labels'] = df['labels'].apply(lambda x : str(int(x) - 1))
    df = shuffle(df)

    df.to_csv(train_out, sep='\t', header=None, index=False)
    df = pd.read_csv(train_out, delimiter='\t', header=None)
    df.to_csv(train_out, sep='\t', header=None)

if __name__ == '__main__':
    set_names = ['./image_tr', './image_dev']

    for name in set_names :
        gen_list(name)

'''
import json
f = open('./image_tr/image_tr.txt', "r")
g = open('train.lst', "w")


result=[]

for line in f:
    #g.write(str(0) + "\t" +"scene_test_a_images_20170922/"+line);
    temp_dict={}

    temp_dict['image_id'] = line[:-1]
    temp_dict['label_id'] = [0]
    result.append(temp_dict)

g.close()
f.close()
'''
