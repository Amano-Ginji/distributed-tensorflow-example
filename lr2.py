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
flags.DEFINE_integer('thread_num', 2, 'thread num')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 120, 'Number of epochs to run trainer.')
flags.DEFINE_integer('batch_size', 500, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('features', 4762348, 'Feature size')
flags.DEFINE_string('train', 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train/part-00000', 'train file')
flags.DEFINE_string('test', 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/test/part-00000', 'test file')
flags.DEFINE_string('checkpoint', 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/checkpoint', 'checkpoint file')
flags.DEFINE_string('train_file_list', 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train_file_list', 'train file list')
flags.DEFINE_string('test_file_list', 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/test_file_list', 'test file list')
flags.DEFINE_integer('trace_step_interval', 10000, 'number of steps to output info')


def debug(msg):
    tf.logging.debug(' [%s:%d] %s' % (FLAGS.job_name, FLAGS.task_index, msg))

def info(msg):
    tf.logging.info(' [%s:%d] %s' % (FLAGS.job_name, FLAGS.task_index, msg))

def error(msg):
    tf.logging.error('[%s:%d] %s' % FLAGS.job_name, FLAGS.task_index, msg)


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

    @staticmethod
    def format_samples_sparse(samples):
        labels = []
        fids = []
        fvals = []
        sp_indices = []
        i = 0
        size = len(samples)
        for i in xrange(0, size):
            sample = samples[i]
            labels.append(sample.label)
            fids += sample.indices
            fvals += sample.values
            for index in sample.indices:
                sp_indices.append([i, index])
        return np.reshape(labels, (size, 1)), fids, fvals, sp_indices, size
        

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
                if line == None or len(line) < 2:
                    continue
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

    def Shuffle(self):
        np.random.shuffle(self.train_samples)

    def NextBatch(self, batch_size=100, data_type='train'):
        if self.mode == 'all': # fetch from full queue
            #self.NextBatchAll(batch_size, data_type)
            samples = self.train_samples
            if data_type != 'train':
                samples = self.test_samples
            n = np.shape(samples)[0]
            for s in xrange(0, n, batch_size):
                e = s+batch_size if s+batch_size<n else n
                yield samples[s:e]
        elif self.mode == 'queue': # fetch from partial queue
            self.NextBatchQueue()

    def NextBatchAll(self, batch_size, data_type):
        samples = self.train_samples
        if data_type != 'train':
            samples = self.test_samples
        n = np.shape(samples)[0]
        for s in xrange(0, n, batch_size):
            e = s+batch_size if s+batch_size<n else n
            yield samples[s:e]

    def NextBatchQueue(self):
        pass

    def GetFileList(self, file_list, num_workers, task_index):
        train_file_list_gfile = tf.gfile.GFile(file_list, mode="r")
        lines = train_file_list_gfile.readlines()
        train_file_list_gfile.close()
        return [line.strip() for line in lines[task_index:len(lines):num_workers]]


def Trace(sess, iter_num, step_num, global_step, cross_entropy, labels, num_features, batch_size, sp_indices, fids, fvals):
    global_step_val, cross_entropy_val = sess.run([global_step, cross_entropy], feed_dict = {
        y: labels
        , x_shape: [num_features, batch_size]
        , x_indices: sp_indices
        , x_fids: fids
        , x_fvals: fvals
    })
    info('epoch: {}, local step: {}, global step: {}, loss: {}'.format(iter_num, step_num, global_step_val, cross_entropy_val))


learning_rate = FLAGS.learning_rate
num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
num_features = FLAGS.features
trace_step_interval = FLAGS.trace_step_interval

cluster_conf = json.load(open('cluster_conf.json', "r"))
cluster_spec = tf.train.ClusterSpec(cluster_conf)
num_workers = len(cluster_conf['worker'])

server = tf.train.Server(cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.job_name == 'ps':
    info('start ...')
    server.join()

elif FLAGS.job_name == 'worker':
    info('start ...')
    is_chief = (FLAGS.task_index == 0)
    data_provider = DataProvider(num_workers, FLAGS.task_index, FLAGS.thread_num, 'all')
    data_provider.LoadData()

    with tf.device(tf.train.replica_device_setter(
        worker_device = '/job:worker/task:%d' % FLAGS.task_index
        , cluster = cluster_spec)):

        # global
        global_step = tf.get_variable('global_step', []
            , initializer = tf.constant_initializer(0)
            , trainable = False)

        # input
        with tf.name_scope('input'):
            x_shape = tf.placeholder(tf.int64)
            x_indices = tf.placeholder(tf.int64)
            x_fids = tf.placeholder(tf.int64)
            x_fvals = tf.placeholder(tf.float32)

            sp_fids = tf.SparseTensor(shape=x_shape, indices=x_indices, values=x_fids)
            sp_fvals = tf.SparseTensor(shape=x_shape, indices=x_indices, values=x_fvals)

            y = tf.placeholder(tf.float32, [None, 1])

        # model
        with tf.name_scope('weights'):
            W = tf.Variable(tf.random_normal([num_features, 1]))

        with tf.name_scope('bias'):
            b = tf.Variable(tf.zeros([1]))

        with tf.name_scope('loss'):
            py_x = tf.add(tf.nn.embedding_lookup_sparse(W, sp_fids, sp_fvals, combiner='sum'), b)
            cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(py_x, y))

        with tf.name_scope('train'):
            grad_op = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = grad_op.minimize(cross_entropy, global_step=global_step)

        with tf.name_scope('evaluate'):
            predict_op = tf.nn.sigmoid(py_x)
            auc_op = tf.contrib.metrics.streaming_auc(predict_op, y)

        # summary
        tf.scalar_summary('cost', cross_entropy)
        summary_op = tf.merge_all_summaries()

        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        init_op = tf.global_variables_initializer()

        supervisor = tf.train.Supervisor(is_chief=is_chief, init_op=init, global_step=global_step)

        config = tf.ConfigProto(allow_soft_placement = True)

        info('Start session ...')

    #with supervisor.managed_session(server.target) as sess:
    with supervisor.prepare_or_wait_for_session(server.target, config=config) as sess:
        sess.run(init_op)

        info('Start train ...')
        step_num = 0
        iter_num = 0
        info('num_epochs: %d' % num_epochs)
        while iter_num < num_epochs:
            data_provider.Shuffle()
            for batch_samples in data_provider.NextBatch(batch_size=100, data_type='train'):
                if batch_samples == None or len(batch_samples) <= 0:
                    break
                labels, fids, fvals, sp_indices, batch_size = Sample.format_samples_sparse(batch_samples)
                _ = sess.run([train_op], feed_dict = {
                    y: labels
                    , x_shape: [num_features, batch_size]
                    , x_indices: sp_indices
                    , x_fids: fids
                    , x_fvals: fvals
                })
                step_num += 1
                if step_num % trace_step_interval == 0:
                    Trace(sess, iter_num, step_num, global_step, cross_entropy, labels, num_features, batch_size, sp_indices, fids, fvals)
            iter_num += 1
            Trace(sess, iter_num, step_num, global_step, cross_entropy, labels, num_features, batch_size, sp_indices, fids, fvals)
            
        info('Finish train.')

        info('Start evaluate ...')
        auc_val = None
        for batch_samples in data_provider.NextBatch(batch_size=100, data_type='test'):
            if batch_samples == None or len(batch_samples) <= 0:
                break
            labels, fids, fvals, sp_indices, batch_size = Sample.format_samples_sparse(batch_samples)
            auc_val = sess.run(auc_op, feed_dict = {
                y: labels
                , x_shape: [num_features, batch_size]
                , x_indices: sp_indices
                , x_fids: fids
                , x_fvals: fvals
            })
        info('Finish evaluate, auc: {}'.format(auc_val))

