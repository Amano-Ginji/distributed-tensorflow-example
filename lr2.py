#coding=utf-8

import numpy as np
import sys
import json
import re

import threading

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

tf.logging.set_verbosity(tf.logging.DEBUG)

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('job_name', 'worker', 'job name')
flags.DEFINE_integer('task_index', 0, 'task index')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 120, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 500, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('features', 4762348, 'Feature size')
flags.DEFINE_string('train', 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train/part-00000', 'train file')
flags.DEFINE_string('test', 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/test/part-00000', 'test file')
flags.DEFINE_string('checkpoint', 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/checkpoint', 'checkpoint file')
flags.DEFINE_string('train_file_list', 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train_file_list', 'train file list')
flags.DEFINE_string('test_file_list', 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/test_file_list', 'test file list')


cluster_conf = json.load(open('cluster_conf.json', "r"))
cluster_spec = tf.train.ClusterSpec(cluster_conf)
num_workers = len(cluster_conf['worker'])

server = tf.train.Server(cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)


def debug(msg):
    tf.logging.debug('%s: %d, msg: %s' % (FLAGS.job_name, FLAGS.task_index, msg))

def info(msg):
    tf.logging.info('%s: %d, msg: %s' % (FLAGS.job_name, FLAGS.task_index, msg))

def error(msg):
    tf.logging.error('%s: %d, msg: %s' % FLAGS.job_name, FLAGS.task_index, msg)


class Sample:
    def __init__(self):
        self.label = -1
        self.indices = []
        self.values = []

    def parse_line_libsvm(self, line):
        line = re.split('[ \t]', line)
        self.label = int(line[0])
        self.indices = []
        self.fvalues = []
        for item in line[1:]:
            [index, value] = item.split(':')
            index = int(index)
            value = float(value)
            self.indices.append(index)
            self.values.append(value)


class LoadDataThread(threading.Thread):
    def __init__(self, tid, files):
        threading.Thread.__init__(self)
        self.samples = []
        self.tid = tid
        self.files = files

    def run(self):
        for file_name in self.files:
            info('thread: %d, input file: %s' % (self.tid, file_name))
            lines_num = 0
            # TODO(yaowq): rewrite to batch generator mode
            for line in tf.gfile.GFile(file_name, mode='r'):
                sample = Sample()
                sample.parse_line_libsvm(line)
                self.samples.append(sample)
                lines_num += 1
            info("thread: %d, input file: %s, samples: %d" % (self.tid, file_name, lines_num))
        info("thread: %d, samples: %d" % (self.tid, len(self.samples)))
            

class DataProvider:
    def __init__(self, num_workers, task_index, thread_num, mode='all'):
        self.mode = mode
        self.num_workers = num_workers
        self.thread_num = thread_num

        self.train_samples = []
        self.test_samples = []
        self.train_file_list = self.GetFileList(FLAGS.train_file_list, num_workers, task_index)
        self.test_file_list = self.GetFileList(FLAGS.test_file_list, num_workers, task_index)
        tf.logging.info('')

        self.train_threads = self.InitThreads(self.train_file_list, len(self.train_file_list), self.thread_num)
        self.test_threads = self.InitThreads(self.test_file_list, len(self.test_file_list), self.thread_num)

    def GetTrainSamples(self):
        return self.train_samples

    def GetTestSamples(self):
        return self.test_samples

    def InitThreads(self, files, file_num, thread_num):
        return [LoadDataThread(tid, files[tid:file_num:thread_num]) for tid in xrange(0, thread_num)]

    def LoadData(self):
        if self.mode == 'all':
            self.LoadDataAll(self.train_samples, self.train_threads)
            self.LoadDataAll(self.test_samples, self.test_threads)
        elif self.mode == 'queue':
            self.LoadDataQueue()

    def LoadDataAll(self, samples, threads):
        for t in threads:
            t.setDaemon(True)
            t.start()
        for t in threads:
            t.join()
        for t in threads:
            samples.extend(t.samples)
        info("samples: %d" % len(samples))

    def LoadDataQueue(self):
        pass

    def NextBatch(self):
        if self.mode == 'all':
            return self.NextBatchAll()
        elif self.mode == 'queue':
            return self.NextBatchQueue()

    def NextBatchAll(self):
        pass

    def NextBatchQueue(self):
        pass

    def GetFileList(self, file_list, num_workers, task_index):
        train_file_list_gfile = tf.gfile.GFile(file_list, mode="r")
        lines = train_file_list_gfile.readlines()
        train_file_list_gfile.close()
        return [line.strip() for line in lines[task_index:len(lines):num_workers]]


