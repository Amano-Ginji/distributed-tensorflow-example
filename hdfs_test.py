import tensorflow as tf
import threading

class LoadDataThread(threading.Thread):
    def __init__(self, id, file_name):
        threading.Thread.__init__(self)
        self.id = id
        self.file_name = file_name
    
    def run(self):
        line_num = 0
        for line in tf.gfile.GFile(self.file_name, mode='r'):
            line_num += 1
        print('thread: %d, lines: %d, file: %s' % (self.id, line_num, self.file_name))
    
file_names = [
    'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train/part-00000',
    'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train/part-00001',
    'hdfs://localhost:9000/user/yaowq/tensorflow/lr/data/train/part-00002'
]
threads = [LoadDataThread(i, file_names[i]) for i in xrange(0, len(file_names))]

for t in threads:
    t.setDaemon(True)
    t.start()
    
for t in threads:
    t.join()
