#coding=utf-8

import unittest
import tensorflow as tf
from lr2 import DataProvider

flags = tf.app.flags
FLAGS = flags.FLAGS


class TestDataProvider(unittest.TestCase):
    def setUp(self):
        FLAGS.train_file_list = 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train_file_list'
        FLAGS.test_file_list = 'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/test_file_list'

    def tearDown(self):
        FLAGS.train_file_list = ''
        FLAGS.test_file_list = ''

    def testInit(self):
        data_provider = DataProvider(1, 0, 2, mode='all')
        data_provider.LoadData()
        self.assertEqual(len(data_provider.GetTrainSamples()), 685)

if __name__ == '__main__':
    unittest.main()


