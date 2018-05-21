'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: configuration.py
@time: 2018/5/16 下午1:33
@desc: shanghaijiaotong university
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """Sets the default model hyperparameters."""
        # File pattern of sharded TFRecord file containing SequenceExample protos.
        # Must be provided in training and evaluation modes.
        self.input_file_pattern = r'D:\Users\XIONGFEI149\xiongfei149\logo\logo相似\code\201859\data_tfrecords\train-?????-of-00001'

        # Image format ("jpeg" or "png").
        self.image_format = "png"

        # Approximate number of values per input shard. Used to ensure sufficient
        # mixing between shards in training.
        self.values_per_input_shard = 2300
        # Minimum number of shards to keep in the input queue.
        self.input_queue_capacity_factor = 2
        # Number of threads for prefetching SequenceExample protos.
        self.num_input_reader_threads = 1
        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4

        # Name of the SequenceExample context feature containing image data.
        self.image_feature_name = "image/data"
        # Name of the SequenceExample feature list containing integer captions.
        self.label_feature_name = "image/image_label"
        self.filename_feature_name = "image/image_name"
        # Batch size.
        self.batch_size = 12
        # image size
        self.image_height = 224
        self.image_width = 224
        self.num_inference_examples = 73662

