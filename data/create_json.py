#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PIL import Image
import os
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

test_root = './image_dev'
label_raw = []

def create_json(lst):

    with open(lst) as fp:
        line = fp.readline();
        line = fp.readline();
        while line:
            l_array = line.rstrip().split(",")
            #print(l_array)

            label_raw.append({'image_id':l_array[0], 'label_id':l_array[1]})
            line = fp.readline();

create_json("image_dev.txt")

with open('./dev_annotations.json', 'w') as f:
    json.dump(label_raw, f)

