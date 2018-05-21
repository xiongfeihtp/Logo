'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: task20180413_img_cluster_ed3.py
@time: 2018/5/16 下午1:28
@desc: shanghaijiaotong university
'''
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import base64
import sys

def encode_array(x):
    res = str(base64.b64encode(x),encoding='utf-8')
    return res

def decode_array(x):
    temp = base64.b64decode(bytes(x,encoding='utf-8'))
    res = np.fromstring(temp,np.float32)
    return res

# feature = pd.read_csv(r'D:\Users\zhouyuan346\Desktop\zhouyuan\2018-03-26\ebd_qcc_logo_feature.csv',index_col=False,encoding='utf8',sep=';')
feature = pd.read_csv(r'/data/zhouyuan346/feature/feature_all.csv', index_col=False, encoding='utf8')
print('load feature')

feature['img_array'] = feature.feature.apply(lambda x: decode_array(x))
print('Decode feature')

n = len(feature)
X = np.zeros((n,feature.img_array[0].shape[0]),dtype=np.float32)
for i in range(n):
    X[i] = feature.img_array[i]
na_index = np.sum(np.isnan(X), axis=1) > 0

X = X[~na_index,:]

# model = DBSCAN(eps=np.float(sys.argv[1]), min_samples=np.int(sys.argv[2]), metric='cosine')
# normalization
Normalize = StandardScaler()
X_normed = Normalize.fit_transform(X)
print('Normalize...')
# PCA
model_PCA = PCA(n_components=500)
X_PCA = model_PCA.fit_transform(X_normed)
print('PCA dimension reduction...')
explained_variance = np.sum(model_PCA.explained_variance_ratio_)
print('explianed variance: %f'%explained_variance)
if explained_variance < 0.9:
    print('PCA under-fit!')
else:
    model = KMeans(n_clusters=np.int(sys.argv[2]))
    # temp = """{'n_clusters':160}"""
    # model = KMeans(**eval(temp))
    label = model.fit_predict(X)
    feature['label_' + sys.argv[1]] = np.nan
    feature['label_' + sys.argv[1]][~na_index] = label
    # feature['label_' + sys.argv[1]][na_index]
    print(feature['label_' + sys.argv[1]].value_counts())
    target_ls = list(feature.columns)
    target_ls.remove('img_array')
    feature = feature[target_ls]
    feature.to_csv('/data/zhouyuan346/feature/feature_labeled.csv',index=False,encoding='utf8')
    print('Finish!')

