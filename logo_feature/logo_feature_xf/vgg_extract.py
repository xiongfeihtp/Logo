'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: vgg_extract.py
@time: 2018/5/16 下午1:40
@desc: shanghaijiaotong university
'''

import numpy as np
import cv2
import os
import sys
import tensorflow as tf
import math
from tqdm import tqdm
from tensorflow.contrib import slim

sys.path.append(r'D:\Users\XIONGFEI149\xiongfei149\logo\logo相似\code\201859')
import vgg16

os.chdir(r'D:\Users\XIONGFEI149\xiongfei149\logo\logo相似\code\201859')
from Input_Mode import Input_Mode
from configuration import ModelConfig
import re

with tf.device('/cpu:0'):
    config = ModelConfig()
    data_model = Input_Mode(config, 'eval')
    data_model.build()
    vgg = vgg16.Vgg16()
    vgg.build(data_model.images)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        queue_runner = tf.train.start_queue_runners(sess=sess, coord=coord)
        num_inference_batches = int(
            math.ceil(config.num_inference_examples / config.batch_size))
        feature = np.zeros((config.num_inference_examples, 4096))
        for i in tqdm(range(num_inference_batches)):
            image_name_list, fc7 = sess.run([data_model.image_names, vgg.output_relu7])
            for name, img in zip(image_name_list, fc7):
                index = int(re.findall("(\d+).png", name.decode())[0])
                feature[str(index)] = img.squeeze()
        coord.request_stop()
        coord.join(queue_runner)
        print("finish")
np.save(r'D:\Users\XIONGFEI149\xiongfei149\logo\logo相似\code\201859\feature\features_vgg16.npy', feature)
