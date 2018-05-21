'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: Input_Mode.py
@time: 2018/5/16 下午1:38
@desc: shanghaijiaotong university
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from ops import image_processing
from ops import inputs as input_ops


class Input_Mode:
    def __init__(self, config, mode):
        assert mode in ["train", "eval"]
        self.config = config
        self.mode = mode

        self.config = config
        # Reader for the input data.
        self.reader = tf.TFRecordReader()
        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = None

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def process_image(self, encoded_image, thread_id=0):
        """Decodes and processes an image string.

        Args:
          encoded_image: A scalar string Tensor; the encoded image.
          thread_id: Preprocessing thread id used to select the ordering of color
            distortions.

        Returns:
          A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(encoded_image,
                                              resize_height=self.config.image_height,
                                              resize_width=self.config.image_width,
                                              thread_id=thread_id,
                                              image_format=self.config.image_format)

    def build_inputs(self):
        # Prefetch serialized SequenceExample protos.
        input_queue = input_ops.prefetch_input_data(
            self.reader,
            self.config.input_file_pattern,
            self.is_training(),
            batch_size=self.config.batch_size,
            values_per_shard=self.config.values_per_input_shard,
            # approximate values nums for all shard
            input_queue_capacity_factor=self.config.input_queue_capacity_factor,
            # queue_capacity_factor for shards
            num_reader_threads=self.config.num_input_reader_threads)

        # Image processing and random distortion. Split across multiple threads
        # with each thread applying a slightly different color distortions.
        assert self.config.num_preprocess_threads % 2 == 0
        images_and_label = []
        for thread_id in range(self.config.num_preprocess_threads):
            # thread
            serialized_sequence_example = input_queue.dequeue()
            encoded_image, image_label, image_name = input_ops.parse_sequence_example(
                serialized_sequence_example,
                image_feature=self.config.image_feature_name,
                label_feature=self.config.label_feature_name,
                filename_feature=self.config.filename_feature_name)
            # preprocessing, for different thread_id use different distortion function
            image = self.process_image(encoded_image, thread_id=thread_id)
            images_and_label.append([image, image_name])
            # mutil threads preprocessing the image

        queue_capacity = (2 * self.config.num_preprocess_threads *
                          self.config.batch_size)

        images, image_names = tf.train.batch_join(
            images_and_label,
            batch_size=self.config.batch_size,
            capacity=queue_capacity,
            dynamic_pad=True,
            name="batch")

        self.images = images
        self.image_names = image_names

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
