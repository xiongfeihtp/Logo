'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: input_handler.py
@time: 2018/5/16 下午1:34
@desc: shanghaijiaotong university
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
from datetime import datetime
import os.path
import random
import sys
import threading
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string("train_image_dir", r'D:\Users\XIONGFEI149\xiongfei149\logo\logo_dataset\picture_all',
                       "Training image directory.")

tf.flags.DEFINE_string("output_dir", "./feature/feature_hoc", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 8,
                        "Number of shards in training TFRecord files.")

tf.flags.DEFINE_integer("num_threads", 4,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_label", "filename"])


class ImageDecoder:
    """Helper class for decoding images in TensorFlow."""

    def __init__(self, function):
        self.function = function

    def decode_vector(self, img):
        vector = self.function(img)
        return vector


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(image, decoder):
    try:
        encoded_vector = decoder.decode_vector(image.filename)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image.filename)
        return

    context = tf.train.Features(feature={
        "image/image_label": _int64_feature(image.image_label),
        "image/data": _bytes_feature(encoded_vector.tobytes()),
        "image/image_name": _bytes_feature((image.filename).encode())
    })

    sequence_example = tf.train.SequenceExample(
        context=context)
    return sequence_example


def _process_image_files(thread_index, ranges, name, images, decoder,
                         num_shards):
    """Processes and saves a subset of images as TFRecord files in one thread.

    Args:
      thread_index: Integer thread identifier within [0, len(ranges)].
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      decoder: An ImageDecoder object.
      vocab: A Vocabulary object.
      num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]
            sequence_example = _to_sequence_example(image, decoder)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1
            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()
        writer.close()
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, images, num_shards, function):
    # # Shuffle the ordering of images. Make the randomization repeatable.
    # random.seed(12345)
    # random.shuffle(images)
    # # Break the images into num_threads batches. Batch i is defined as
    # # images[ranges[i][0]:ranges[i][1]].

    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    # give the range for different threads
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()
    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder(function)
    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, images, decoder, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)
    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
          (datetime.now(), len(images), name))


def data_save_as_tfrecord(function):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    dataset = []
    list = os.listdir(FLAGS.train_image_dir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(FLAGS.train_image_dir, list[i])
        # 这里没有label，设置为零
        dataset.append(ImageMetadata(0, path))
    train_dataset = dataset
    # give the tfrecords for multi-threads
    _process_dataset("HOC", train_dataset, FLAGS.train_shards, function)
