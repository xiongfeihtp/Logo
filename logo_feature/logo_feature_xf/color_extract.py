'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: color_extract.py
@time: 2018/5/16 下午1:31
@desc: shanghaijiaotong university
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
from tqdm import tqdm

sys.path.append(r'D:\Users\XIONGFEI149\xiongfei149\logo\logo相似\code\201859')
import os
import json

os.chdir(r'D:\Users\XIONGFEI149\xiongfei149\logo\logo相似\code\201859')
from color_feature import ColorDescriptor

os.chdir(r'D:\Users\XIONGFEI149\xiongfei149\logo\logo_dataset\picture_all')
name = os.listdir(r'.')
img = cv2.imread(name[0])
cd = ColorDescriptor((12, 12, 12))
if img.shape != (224, 224, 3):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
hoc = cd.describe(img)
print("img numbers: {}, feature_dim: {}".format(len(name), len(hoc)))
img_index_list = range(len(name))
feature = np.zeros((len(name), len(hoc)))
for i in tqdm(img_index_list):
    img_name = '{}.png'.format(i)
    image = cv2.imread(img_name)
    if image.shape != (224, 224, 3):
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    feature[i] = cd.describe(img)
save_path = r"D:\Users\XIONGFEI149\xiongfei149\logo\logo相似\code\201859\feature\features_color.txt"
np.save(save_path, feature)
print("finish...")
