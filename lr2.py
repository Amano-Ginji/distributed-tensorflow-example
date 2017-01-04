#coding=utf-8

import numpy as np
import sys
import json
import re

import multiprocessing
from multiprocessing import cpu_count

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


class LoadDataProcess(multiprocessing.Process):
    def __init__(self, pid, files, thread_num=4):
        multiprocessing.Process.__init__(self)
        self.samples = []
        self.pidx = pid
        self.files = files
        self.thread_num = thread_num
        self.threads = self.InitThreads(files, len(files), thread_num)

    def InitThreads(self, files, file_num, thread_num):
        return [LoadDataThread(self.pidx, tid, files[tid:file_num:thread_num]) for tid in xrange(0, thread_num)]

    def run(self):
        for t in self.threads:
            t.setDaemon(True)
            t.start()
        for t in self.threads:
            t.join()
        for t in self.threads:
            self.samples.extend(t.samples)
        tf.logging.info("pid: %d, samples: %d", self.pidx, len(self.samples))


class LoadDataThread(threading.Thread):
    def __init__(self, pid, tid, files):
        threading.Thread.__init__(self)
        self.samples = []
        self.pidx = pid
        self.tid = tid
        self.files = files

    def run(self):
        for file_name in self.files:
            tf.logging.info("pid: %d, thread: %d, input file: %s", self.pidx, self.tid, file_name)
            lines_num = 0
            input_file = tf.gfile.GFile(file_name, mode='r')
            print input_file
            input_file.readline()
            '''
            # TODO(yaowq): rewrite to batch generator mode
            #for line in input_file.readlines(): # iterator
            #for line in input_file: # generator
            for line in input_file.readlines():
                print "hello"
                #sample = Sample()
                #sample.parse_line_libsvm(line)
                #self.samples.append(sample)
                lines_num += 1
                if lines_num == 1:
                    break
            '''
            tf.logging.info("pid: %d, thread: %d, input file: %s, samples: %d", self.pidx, self.tid, file_name, lines_num)
        tf.logging.info("pid: %d, thread: %d, samples: %d", self.pidx, self.tid, len(self.samples))
            

class DataProvider:
    def __init__(self, num_workers, task_index, mode='all'):
        self.mode = mode
        self.num_workers = num_workers
        self.file_list = self.GetFileList(FLAGS.train_file_list, num_workers, task_index)
        self.procs = self.InitProcs(self.file_list, len(self.file_list), cpu_count())

        # input placeholder
        self.x_shape = tf.placeholder(tf.int64)
        self.x_indices = tf.placeholder(tf.int64)
        self.x_fids = tf.placeholder(tf.int64)
        self.x_fvals = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.int8, [None, 1])

        self.sp_fids = tf.SparseTensor(shape=self.x_shape, indices=self.x_indices, values=self.x_fids)
        self.sp_fvals = tf.SparseTensor(shape=self.x_shape, indices=self.x_indices, values=self.x_fvals)

        # data
        if mode == "all":
            self.train_data = self.LoadData()
            #self.test_data = self.LoadData()
        elif mode == "queue":
            self.train_data_queue = self.LoadDataQueue()
            self.test_data_queue = self.LoadDataQueue()

    def InitProcs(self, file_list, file_num, cpu_num):
        return [LoadDataProcess(cpu_idx, file_list[cpu_idx:file_num:cpu_num], 1) for cpu_idx in xrange(0, cpu_num)]

    def LoadData(self):
        samples = []
        for p in self.procs:
            p.daemon = True
            p.start()
        for p in self.procs:
            p.join()
        for p in self.procs:
            samples.extend(p.samples)
        return samples

    '''
    def LoadData(self):
        ps = []
        file_num = len(self.file_list)
        cpu_num = cpu_count()
        for cpu_idx in range(0, cpu_num):
            proc = multiprocessing.Process(target=LoadDataWorker, args=(self.file_list[cpu_idx:file_num:cpu_num],))
            ps.append(proc)
        for p in ps:
            proc.daemon = True
            p.start()
        for p in ps:
            p.join()

    def LoadDataWorker(self, files):
        thread_num = 4
        file_num = len(files)
        threads = []
        for tid in range(0, thread_num):
            thread = threading.Thread(target=LoadDataTask, args=(files[tid:file_num:thread_num],))
            threads.append(thread)
        for t in threads:
            t.setDaemon(True)
            t.start()
        for t in threads:
            t.join()

    def LoadDataTask(self, files):
        for file_name in files:
            input_file = tf.gfile.GFile(file_name, mode='r')
            # TODO(yaowq): rewrite to batch generator mode
            #for line in input_file.readlines():
            for line in input_file.readline():
                self.instances.append(parse_line_libsvm(line))

    def parse_line_libsvm(self, line):
        line = re.split('[ \t]', line)
        label = int(line[0])
        indices = []
        values = []
        for item in line[1:]:
            [index, value] = item.split(':')
            index = int(index)
            value = float(value)
            indices.append(index)
            values.append(value)
        return label, indices, values
    '''

    def LoadDataQueue(self):
        pass

    def NextBatch(self):
        pass

    def GetFileList(self, train_file_list, num_workers, task_index):
        train_file_list_gfile = tf.gfile.GFile(train_file_list, mode="r")
        lines = train_file_list_gfile.readlines()
        train_file_list_gfile.close()
        return lines[task_index:len(lines):num_workers]

cluster_conf = json.load(open('cluster_conf.json', "r"))
cluster_spec = tf.train.ClusterSpec(cluster_conf)

num_workers = len(cluster_conf['worker'])

server = tf.train.Server(cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)


'''
if FLAGS.job_name == 'ps':
    print('ps-{} start ...'.format(FLAGS.task_index))
    server.join()

elif FLAGS.job_name == 'worker':
    print('worker-{} start ...'.format(FLAGS.task_index))
    is_chief = (FLAGS.task_index == 0)
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster_spec)):
        # global
        global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), trainable = False)

        # data
        x_shape = tf.placeholder(tf.int64)
        x_indices = tf.placeholder(tf.int64)
        x_fids = tf.placeholder(tf.int64)
        x_fvals = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.int8, [None, 1])

        sp_fids = tf.SparseTensor(shape=x_shape, indices=x_indices, values=x_fids)
        sp_fvals = tf.SparseTensor(shape=x_shape, indices=x_indices, values=x_fvals)

        # model
        W = init_weights([FLAGS.features, 1])
        b = init_weights([1])
        py_x = tf.add(tf.nn.embedding_lookup_sparse(W, sp_fids, sp_fvals, combiner="sum"),b)
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(py_x, y))
        predict_op = tf.nn.sigmoid(py_x)

        # optimizer
        opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_op = opt.minimize(cost, global_step=global_step)

        # eval
        auc_op = tf.contrib.metrics.streaming_auc(predict_op, y)

        # init
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(is_chief=is_chief, init_op=init, global_step=global_step)

        config = tf.ConfigProto(allow_soft_placement=True)
        logging.info('Start waiting/prepare for session.')

    with sv.managed_session(server.target) as sess:
        sess.run(init_op)

        iter_num = 0
        while iter_num < FLAGS.epochs:
            = next_batch()
            while 


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w):
    return tf.matmul(X, w, a_is_sparse=True)

def read_batch(sess, train_data, batch_size):
    label_list = []
    ids = []
    sp_indices = []
    weight_list = []
    for i in xrange(0, batch_size):
        try:
            line = sess.run(train_data)
        except tf.errors.OutOfRangeError as e:
            logging.info("All data read finished.")
            return np.reshape(label_list, (i, 1)), ids, sp_indices, weight_list, i
        label, indices, values = parse_line_for_batch_for_libsvm(line)
        label_list.append(label)
        ids += indices
        for index in indices:
            sp_indices.append([i, index])
        weight_list += values
    return np.reshape(label_list, (batch_size, 1)), ids, sp_indices, weight_list, batch_size


test_gfile = tf.gfile.GFile(FLAGS.test, mode="r")
def read_train_batch_with_gfile(train_gfile, batch_size):
    label_list = []
    ids = []
    sp_indices = []
    weight_list = []
    for i in xrange(0, batch_size):
        try:
            line = train_gfile.readline() 
            if line == None:
                return np.reshape(label_list, (i, 1)), ids, sp_indices, weight_list, i, train_gfile
            if len(line) < 2:
                return np.reshape(label_list, (i, 1)), ids, sp_indices, weight_list, i, train_gfile
        except tf.errors.OutOfRangeError as e:
            logging.info("All data read finished.")
            return np.reshape(label_list, (i, 1)), ids, sp_indices, weight_list, i
        label, indices, values = parse_line_for_batch_for_libsvm(line)
        label_list.append(label)
        ids += indices
        for index in indices:
            sp_indices.append([i, index])
        weight_list += values
    return np.reshape(label_list, (batch_size, 1)), ids, sp_indices, weight_list, batch_size, train_gfile




def read_test_batch_with_gfile(batch_size):
    label_list = []
    ids = []
    sp_indices = []
    weight_list = []
    for i in xrange(0, batch_size):
        try:
            line = test_gfile.readline() 
            if line == None:
                return np.reshape(label_list, (i, 1)), ids, sp_indices, weight_list, i
            if len(line) < 2:
                return np.reshape(label_list, (i, 1)), ids, sp_indices, weight_list, i
        except tf.errors.OutOfRangeError as e:
            logging.info("All data read finished.")
            return np.reshape(label_list, (i, 1)), ids, sp_indices, weight_list, i
        label, indices, values = parse_line_for_batch_for_libsvm(line)
        label_list.append(label)
        ids += indices
        for index in indices:
            sp_indices.append([i, index])
        weight_list += values
    return np.reshape(label_list, (batch_size, 1)), ids, sp_indices, weight_list, batch_size


def parse_line_for_batch_for_libsvm(line):
    line = re.split('[ \t]', line)
    label = int(line[0])
    indices = []
    values = []
    for item in line[1:]:
        [index, value] = item.split(':')
        index = int(index)
        value = float(value)
        indices.append(index)
        values.append(value)
    return label, indices, values

learning_rate = FLAGS.learning_rate
num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
num_features = FLAGS.features
trainset_file = FLAGS.train.split(',')
testset_file = FLAGS.test

cluster_conf = json.load(open("cluster_conf.json", "r"))
cluster_spec = tf.train.ClusterSpec(cluster_conf)

num_workers = len(cluster_conf['worker'])

server = tf.train.Server(cluster_spec, job_name=FLAGS.job_name, task_index = FLAGS.task_index)



if FLAGS.job_name == "ps":
    print("ps {} start".format(FLAGS.task_index))

    server.join()

elif FLAGS.job_name == "worker" :
    print("worker {} start".format(FLAGS.task_index))

    #run_training(server, cluster_spec, num_workers)
    is_chief = (FLAGS.task_index == 0)
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/replica:0/task:%d/cpu:0" % FLAGS.task_index, cluster = cluster_spec)) :
        global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0), trainable = False)

        X = tf.placeholder("float", [None, num_features]) # create symbolic variables

        sp_indices = tf.placeholder(tf.int64)
        sp_shape = tf.placeholder(tf.int64)
        sp_ids_val = tf.placeholder(tf.int64)
        sp_weights_val = tf.placeholder(tf.float32)

        sp_ids = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
        sp_weights = tf.SparseTensor(sp_indices, sp_weights_val, sp_shape)

        Y = tf.placeholder(tf.float32, [None, 1])

        W = init_weights([num_features, 1])

        py_x = tf.nn.embedding_lookup_sparse(W, sp_ids, sp_weights, combiner="sum")

        # 增加两层神经网络
        #deep_w1 = init_weights([num_features, 128])
        #deep_h1 = tf.nn.embedding_lookup_sparse(deep_w1, sp_ids, sp_weights, combiner="sum")
        #deep_w2 = init_weights([128, 1])
        #deep_h2 = tf.matmul(deep_h1,deep_w2)
        #py_x += deep_h2

        logits = tf.nn.sigmoid_cross_entropy_with_logits(py_x, Y)

        cost = tf.reduce_sum(logits)

        opt = tf.train.GradientDescentOptimizer(learning_rate)
        #opt = tf.train.SyncReplicasOptimizerV2(opt, replicas_to_aggregate=num_workers, total_num_replicas=num_workers)

        train_step = opt.minimize(cost, global_step=global_step)

        predict_op = tf.nn.sigmoid(py_x)

        auc_op = tf.contrib.metrics.streaming_auc(predict_op, Y)

        #init_token_op = opt.get_init_tokens_op()
        #chief_queue_runner = opt.get_chief_queue_runner()

        #init = [tf.initialize_all_variables(), tf.initialize_local_variables()]
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(is_chief=is_chief, init_op=init, global_step=global_step)

        config = tf.ConfigProto(allow_soft_placement=True)
        logging.info('Start waiting/prepare for session.')
        #sess = sv.prepare_or_wait_for_session(server.target, config=config)
    with sv.managed_session(server.target) as sess:
        sess.run(init_op)

        #if is_chief:
        #    logging.info('Before start queue runners.')
        #    sv.start_queue_runners(sess, [chief_queue_runner])
        #    logging.info('Start queue runners success.')
        #    sess.run(init_token_op)
        #    logging.info('Run init tokens op success.')

        step = 0
        iter_num = 0
        while iter_num < 100:
            train_gfile = tf.gfile.GFile(FLAGS.train, mode="r")
            logging.info("local iter is %d" %(iter_num))
            while True:
                #logging.info("local step is %d" %(step))
                label, indices, sparse_indices, weight_list, read_count ,train_gfile= read_train_batch_with_gfile(train_gfile, batch_size)
                if read_count == 0:
                    break
                if step % 100 == 0:
                    global_step_val = sess.run(global_step)
                    logging.info('Current step is {}, global step is {}'.format(step, global_step_val))
                sess.run(train_step, feed_dict = { Y: label, sp_indices: sparse_indices, sp_shape: [num_features, read_count], sp_ids_val: indices, sp_weights_val: weight_list })
                step += 1
                if read_count < batch_size:
                    logging.info('All data trained finished. Last batch size is: {}'.format(read_count))
                    break
            iter_num = iter_num + 1

        auc_value = None
        logging.info("finish to train model")
        while True:
            label, indices, sparse_indices, weight_list, read_count = read_test_batch_with_gfile(batch_size)
            if read_count == 0:
                break
            auc_value = sess.run(auc_op, feed_dict = { Y: label, sp_indices: sparse_indices, sp_shape: [num_features, read_count], sp_ids_val: indices, sp_weights_val: weight_list })
            if read_count < batch_size:
                break
        logging.info('AUC is {}'.format(auc_value))
'''
