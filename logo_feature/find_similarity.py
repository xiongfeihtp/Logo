import json
import numpy as np
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt

with open('./vector_dic','r') as f:
    vector_dic=json.load(f)
def moreCos(a,b):
    sum_fenzi = 0
    sum_fenmu = 1
    for i in range(len(a)):
        sum_fenzi += a[i]*b[i]
        sum_fenmu *= np.sqrt(a[i]**2 + b[i]**2 )
    return sum_fenzi/sum_fenmu

path='./picture_place'
top_k=10
str = input("请输入图片索引：")
src_img_name=str+'.png'
src_img_path=path+'/'+str+'.png'
distance_dic={}
src_array=vector_dic[src_img_name]
for item in vector_dic:
    if item is not src_img_name:
        distance_dic[item]=moreCos(src_array,vector_dic[item])
distance_dic_sorted = OrderedDict(sorted(distance_dic.items(), key=lambda t: t[1]))
for i,item in enumerate(distance_dic_sorted):
    if i is not top_k:
        item_path=path+'/'+item
        src_img = cv2.imread(src_img_path)
        item_img =cv2.imread(item_path)
        plt.subplot(121)
        plt.imshow(src_img)
        plt.title('morecos:{}'.format(distance_dic_sorted[item]))
        plt.subplot(122)
        plt.imshow(item_img)
        plt.show()








