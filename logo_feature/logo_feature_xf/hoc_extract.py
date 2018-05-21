'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: hoc_extract.py
@time: 2018/5/16 下午1:33
@desc: shanghaijiaotong university
'''
import cv2
import os
import numpy as np
import sys
from tqdm import tqdm
import json

sys.path.append(r'D:\Users\XIONGFEI149\xiongfei149\logo\logo相似\code\201859')
import os
from input_handler import data_save_as_tfrecord
os.chdir(r'D:\Users\XIONGFEI149\xiongfei149\logo\logo相似\code\201859')
winSize = (32, 32)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = 4
histogramNormType = 0
L2HysThreshold = 2e-1
gammaCorrection = 0
nlevels = 64
winStride = (16, 16)
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
                        L2HysThreshold, gammaCorrection, nlevels)

def Hoc_function(img_name):
    img = cv2.imread(img_name)
    if img.shape != (224, 224, 3):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    h = hog.compute(img, winStride).squeeze()
    return h
"""
save as tfrecored for too large dimension
"""
data_save_as_tfrecord(Hoc_function)
# name = os.listdir(r'.')
# img = cv2.imread(name[0])
# if img.shape != (224, 224, 3):
#     img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
# hoc = hog.compute(img, winStride)
# print("img numbers: {}, feature_dim: {}".format(len(name), len(hoc)))
# vector_dict = {}
# for i in tqdm(range(len(name))[0:2]):
#     img_name = '{}.png'.format(i)
#     image = cv2.imread(img_name)
#     if img.shape != (224, 224, 3):
#         img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
#     h = hog.compute(img, winStride)
#     print(h)
#     vector_dict[str(i)] = h.squeeze().tolist()
# # don't save as npy, json is more efficiency
# save_path = r"D:\Users\XIONGFEI149\xiongfei149\logo\logo相似\code\201859\feature\features_HOG.txt"
# with open(save_path, 'w') as f:
#     json.dump(vector_dict, f)
# print("finish")
