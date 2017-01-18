#coding=utf-8

import numpy as np
import sys
import json
import re
import time
import random

import threading

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

tf.logging.set_verbosity(tf.logging.INFO)

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
flags.DEFINE_integer('trace_step_interval', 10000, 'number of steps to output info')
flags.DEFINE_float('train_sampling_rate', 1.0, 'samling rate for train file')
flags.DEFINE_float('test_sampling_rate', 1.0, 'samling rate for test file')
flags.DEFINE_string('mode', 'all', 'load data all or queue')
flags.DEFINE_integer('train_queue_capacity', 2000, 'train queue capacity')
flags.DEFINE_integer('test_queue_capacity', 2000, 'test queue capacity')


def debug(msg):
    tm = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    tf.logging.debug(' [%s] [%s:%d] %s' % (tm, FLAGS.job_name, FLAGS.task_index, msg))

def info(msg):
    tm = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    tf.logging.info(' [%s] [%s:%d] %s' % (tm, FLAGS.job_name, FLAGS.task_index, msg))

def error(msg):
    tm = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    tf.logging.error(' [%s] [%s:%d] %s' % (tm, FLAGS.job_name, FLAGS.task_index, msg))


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
    def __init__(self, tid, files, sampling_rate, mode='all', sess=None, is_train=True):
        threading.Thread.__init__(self)
        self.samples = []
        self.tid = tid
        self.files = files
        self.sampling_rate = sampling_rate
        self.mode = mode
        self.sess = sess
        self.is_train = is_train

    def run(self):
        if self.mode == 'all':
            for file_name in self.files:
                info('thread: %d, input file: %s' % (self.tid, file_name))
                lines_num = 0
                # TODO(yaowq): rewrite to batch generator mode
                for line in tf.gfile.GFile(file_name, mode='r'):
                    if line == None or len(line) < 2 or random.random()<1-self.sampling_rate:
                        continue
                    sample = Sample()
                    sample.parse_line_libsvm(line)
                    self.samples.append(sample)
                    lines_num += 1
                info("thread: %d, input file: %s, samples: %d" % (self.tid, file_name, lines_num))
            info("thread: %d, samples: %d" % (self.tid, len(self.samples)))
        elif self.mode == 'queue':
            total_lines_num = 0
            lines = []
            while True:
                for file_name in self.files:
                    debug('thread: %d, input file: %s' % (self.tid, file_name))
                    lines_num = 0

                    # TODO(yaowq): rewrite to batch generator mode
                    for line in tf.gfile.GFile(file_name, mode='r'):
                        if line == None or len(line) < 2 or random.random()<1-self.sampling_rate:
                            continue
                        
                        if total_lines_num > 0 and total_lines_num % FLAGS.batch_size == 0:
                            if self.is_train:
                                self.sess.run(DataProvider.train_enqueue_op, feed_dict={
                                    DataProvider.train_queue_input: lines
                                })
                                '''
                                self.sess.run(DataProvider.train_data_queue.dequeue())
                                train_sample = self.sess.run(DataProvider.train_dequeue_op)
                                train_sample = self.sess.run(DataProvider.train_data_batch)
                                print("------ len: {}".format(len(train_sample)))
                                print(np.shape(train_sample))
                                print(train_sample)
                                '''
                            else:
                                self.sess.run(DataProvider.test_enqueue_op, feed_dict={
                                    DataProvider.test_queue_input: lines
                                })
                                '''
                                test_sample = self.sess.run(DataProvider.test_data_batch)
                                print("------ len: {}".format(len(test_sample)))
                                print(np.shape(test_sample))
                                print(test_sample)
                                '''
                            del lines[:]

                        lines.append([line])
                        lines_num += 1
                        total_lines_num += 1
                    debug("thread: %d, input file: %s, samples: %d, total samples: %d" % (self.tid, file_name, lines_num, total_lines_num))
            debug("thread: %d, samples: %d" % (self.tid, total_lines_num))
            

class DataProvider:
    train_data_queue = tf.FIFOQueue(capacity=FLAGS.train_queue_capacity, dtypes=[tf.string], shapes=[[1]])
    train_queue_input = tf.placeholder(tf.string, shape=[FLAGS.batch_size, 1])
    train_enqueue_op = train_data_queue.enqueue_many([train_queue_input])
    train_dequeue_op = train_data_queue.dequeue()
    train_data_batch = tf.train.batch([train_dequeue_op], batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size)

    test_data_queue = tf.FIFOQueue(capacity=FLAGS.test_queue_capacity, dtypes=[tf.string], shapes=[[1]])
    test_queue_input = tf.placeholder(tf.string, shape=[FLAGS.batch_size, 1])
    test_enqueue_op = test_data_queue.enqueue_many([test_queue_input])
    test_dequeue_op = test_data_queue.dequeue()
    test_data_batch = tf.train.batch([test_dequeue_op], batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size)

    # queue driver
    coord = None
    threads_dequeue = None

    run_options = tf.RunOptions(timeout_in_ms=1000)

    def __init__(self, num_workers, task_index, thread_num, mode='all'):
        self.mode = mode
        self.num_workers = num_workers
        self.task_index = task_index
        self.thread_num = thread_num

        self.train_samples = []
        self.test_samples = []

        self.train_file_list = []
        self.test_file_list = []

        self.train_threads = None
        self.test_threads = None

        self.sess = None

    def init(self, sess=None):
        self.sess = sess

        self.train_file_list = self.GetFileList(FLAGS.train, self.num_workers, self.task_index)
        self.test_file_list = self.GetFileList(FLAGS.test, self.num_workers, self.task_index)

        self.train_threads = self.InitThreads(self.train_file_list, len(self.train_file_list)
            , self.thread_num, FLAGS.train_sampling_rate, sess)
        self.test_threads = self.InitThreads(self.test_file_list, len(self.test_file_list)
            , self.thread_num, FLAGS.test_sampling_rate, sess) 

        if self.mode == 'queue':
            # start data enqueue threads
            DataProvider.coord = tf.train.Coordinator()
            DataProvider.threads_dequeue = tf.train.start_queue_runners(coord=DataProvider.coord, sess=sess)

    def GetTrainSamples(self):
        return self.train_samples

    def GetTestSamples(self):
        return self.test_samples

    def GetTestSamplesSampled(self, sampling_rate=1.0, sampling_max_num=1000000):
        if self.mode == 'all':
            return Sample.format_samples_sparse(np.random.choice(self.test_samples, size=min(sampling_max_num,
                int(sampling_rate*np.shape(self.test_samples)[0])), replace=False))
        elif self.mode == 'queue':
            samples = []
            for i in range(0, int(sampling_max_num/FLAGS.batch_size)):
                try:
                    #samples.extend(self.NextBatchQueue(data_type='test'))
                    # NOTE(yaowq): should be 'test' !!!!!
                    samples.extend(self.NextBatchQueue(data_type='train'))
                except Exception,e:
                    print("ignore exception: ", e)
                    time.sleep(1)
            return Sample.format_samples_sparse(samples)

    def InitThreads(self, files, file_num, thread_num, sampling_rate=1.0, sess=None):
        return [LoadDataThread(tid, files[tid:file_num:thread_num], sampling_rate, self.mode, sess) for tid in xrange(0, thread_num)]

    def LoadData(self):
        if self.mode == 'all':
            self.LoadDataAll(self.train_samples, self.train_threads)
            self.LoadDataAll(self.test_samples, self.test_threads)
        elif self.mode == 'queue':
            info("start load train data queue ...")
            self.LoadDataQueue(self.train_threads)
            info("start load test data queue ...")
            self.LoadDataQueue(self.test_threads)

    def LoadDataAll(self, samples, threads):
        for t in threads:
            t.setDaemon(True)
            t.start()
        for t in threads:
            t.join()
        for t in threads:
            samples.extend(t.samples)
        info("samples: %d" % len(samples))

    def LoadDataQueue(self, threads):
        for t in threads:
            t.setDaemon(True)
            t.start()

    def Shuffle(self):
        np.random.shuffle(self.train_samples)

    def NextBatch(self, data_type='train'):
        if self.mode == 'all': # fetch from all
            #self.NextBatchAll(data_type)
            samples = self.train_samples if data_type == 'train' else self.test_samples
            n = np.shape(samples)[0]
            for s in xrange(0, n, FLAGS.batch_size):
                e = s+FLAGS.batch_size if s+FLAGS.batch_size<n else n
                yield samples[s:e]
        elif self.mode == 'queue': # fetch from queue
            #return self.NextBatchQueue()
            data_batch = DataProvider.train_data_batch if data_type == 'train' else DataProvider.test_data_batch
            curr_data_batch = self.sess.run(data_batch)
            samples_batch = []
            for line in curr_data_batch[0]:
                sample = Sample()
                sample.parse_line_libsvm(line)
                samples_batch.append(sample) 
            yield samples_batch
            

    def NextBatchAll(self, data_type):
        samples = self.train_samples
        if data_type != 'train':
            samples = self.test_samples
        n = np.shape(samples)[0]
        for s in xrange(0, n, FLAGS.batch_size):
            e = s+FLAGS.batch_size if s+FLAGS.batch_size<n else n
            yield samples[s:e]

    def NextBatchQueue(self, data_type):
        data_batch = DataProvider.train_data_batch if data_type == 'train' else DataProvider.test_data_batch
        curr_data_batch = self.sess.run(data_batch)
        samples_batch = []
        for line in curr_data_batch[0]:
            sample = Sample()
            sample.parse_line_libsvm(line)
            samples_batch.append(sample) 
        return samples_batch

    def GetFileList(self, input_files, num_workers, task_index):
        file_list = input_files.strip(",").split(",")
        return [f.strip() for f in file_list[task_index:len(file_list):num_workers]]


def Test(sess, iter_num, step_num, global_step, cross_entropy, labels, num_features, batch_size, sp_indices, fids, fvals):
    global_step_val, cross_entropy_val = sess.run([global_step, cross_entropy], feed_dict = {
        y: labels
        , x_shape: [num_features, batch_size]
        , x_indices: sp_indices
        , x_fids: fids
        , x_fvals: fvals
    })
    info('epoch: {}, local step: {}, global step: {}, loss: {}'.format(iter_num, step_num, global_step_val, cross_entropy_val))


def main(_):
    learning_rate = FLAGS.learning_rate
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    num_features = FLAGS.features
    trace_step_interval = FLAGS.trace_step_interval
    
    cluster_conf = json.load(open('cluster_conf.json', "r"))
    cluster_spec = tf.train.ClusterSpec(cluster_conf)
    num_workers = len(cluster_conf['worker'])

    sync_queue_name_template = "shared_queue_{}"
    
    server = tf.train.Server(cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    
    if FLAGS.job_name == 'ps':
        info('start ...')
        server.join()

        '''
        sync_queue_name = sync_queue_name_template.format(FLAGS.task_index)
        queue = tf.FIFOQueue(1, tf.int32, shared_name=sync_queue_name)
        dequeue_op = queue.dequeue()
        sess = tf.Session(server.target)
        info("Waiting for workers done, queue: {}".format(sync_queue_name))
        for i in range(0, num_workers):
            sess.run(dequeue_op)
        info("Terminating parameter server: {}".format(FLAGS.task_index))
        '''
    
    elif FLAGS.job_name == 'worker':
        info('start ...')
        is_chief = (FLAGS.task_index == 0)

        data_provider = DataProvider(num_workers, FLAGS.task_index, FLAGS.thread_num, FLAGS.mode)
        if FLAGS.mode == 'all':
            info('load data')
            data_provider.init()
            data_provider.LoadData()

        info('build graph')
        with tf.device(tf.train.replica_device_setter(
            worker_device = '/job:worker/task:%d' % FLAGS.task_index
            , cluster = cluster_spec)):
    
            # global
            global_step = tf.get_variable('global_step', []
                , initializer = tf.constant_initializer(0)
                , trainable = False)
    
            # input
            with tf.name_scope('input'):
                global x_shape,x_indices,x_fids,x_fvals,sp_fids,sp_fvals,y

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
                #cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(py_x, y))
    
            with tf.name_scope('train'):
                grad_op = tf.train.GradientDescentOptimizer(learning_rate)
                train_op = grad_op.minimize(cross_entropy, global_step=global_step)
    
            with tf.name_scope('evaluate'):
                predict_op = tf.nn.sigmoid(py_x)
                auc_op = tf.contrib.metrics.streaming_auc(predict_op, y)
    
            '''
            # summary
            tf.scalar_summary('cost', cross_entropy)
            summary_op = tf.merge_all_summaries()
            '''
    
            init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
            init_op = tf.global_variables_initializer()
    
            supervisor = tf.train.Supervisor(is_chief=is_chief, init_op=init, global_step=global_step)
    
            config = tf.ConfigProto(allow_soft_placement = True)
    
        info('Start session ...')
    
        #with supervisor.managed_session(server.target) as sess:
        with supervisor.prepare_or_wait_for_session(server.target, config=config) as sess:
            sess.run(init_op)

            if FLAGS.mode == 'queue':
                info('load data queue ...')
                data_provider.init(sess)
                data_provider.LoadData()

            info('sampling test data ...')
            global test_data
            test_data = data_provider.GetTestSamplesSampled(sampling_rate=0.1, sampling_max_num=1000)

            info('Start train ...')
            step_num = 0
            iter_num = 0
            info('num_epochs: %d' % num_epochs)
            while iter_num < num_epochs:
                data_provider.Shuffle()
                for batch_samples in data_provider.NextBatch(data_type='train'):
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
                        Test(sess, iter_num, step_num, global_step, cross_entropy, test_data[0], num_features, test_data[4], test_data[3], test_data[1], test_data[2])
                Test(sess, iter_num, step_num, global_step, cross_entropy, test_data[0], num_features, test_data[4], test_data[3], test_data[1], test_data[2])
                iter_num += 1
                
            info('Finish train.')
    
            info('Start evaluate ...')
            auc_val = None
            for batch_samples in data_provider.NextBatch(data_type='test'):
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

        # close data queue and session
        if FLAGS.mode == 'queue':
            sess.run(DataProvider.train_data_queue.close(cancel_pending_enqueues=True))
            sess.run(DataProvider.test_data_queue.close(cancel_pending_enqueues=True))
            DataProvider.coord.request_stop()
            DataProvider.coord.join(DataProvider.threads_dequeue)
        sess.close()


if __name__ == "__main__":
    tf.app.run(main=main)

