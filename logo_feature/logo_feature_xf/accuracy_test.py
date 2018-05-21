'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: accuracy_test.py
@time: 2018/5/16 下午1:31
@desc: shanghaijiaotong university
'''
import json
import numpy as np
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def feature_dict(name, feature):
    temp = zip(name, feature)
    vector_dic1 = {a: b for a, b in temp}
    return vector_dic1

def cosine_dist(x, y):
    num = np.dot(x, y)
    denum = np.linalg.norm(x) * np.linalg.norm(y)
    res = num / denum
    return res

def accuracy_cal(features, test_index_list, label_dict, topk):
    right_count = {k: 0 for k in topk}
    log_f = open('result_vgg19_fc7.log', 'w')
    for index in tqdm(test_index_list):
        query = features[index]
        dis_dict = {}
        for i in range(len(features)):
            if i != index:
                dis_dict[i] = cosine_dist(query, features[i])
        sort_dict = sorted(dis_dict.items(), key=lambda x: x[1], reverse=True)
        for k in topk:
            #int
            pred_list = [item[0] for item in sort_dict[:k]]
            #int
            label_list = label_dict[str(index)]
            intersect = set(pred_list) & set(label_list)
            if len(intersect) > 0:
                print(k, index, label_list, intersect, file=log_f)
                right_count[k] += 1
    for key in right_count:
        right_count[key] = right_count[key] / len(test_index_list)
    log_f.close()
    return right_count
os.chdir(r'D:\Users\XIONGFEI149\xiongfei149\logo\logo相似\code\201859\feature')
test_img_index_list = list(range(70961, 73662))
print("loading labels...")
with open('logo_pair_label.txt', 'r') as f:
    label_dict = json.load(f)

print("topk calculating...")
top_k = [10, 100, 1000]
feature_color = np.load('./features_color.npy')
accuracy_dict = accuracy_cal(feature_color, test_img_index_list, label_dict, top_k)
del feature_color
print(accuracy_dict)