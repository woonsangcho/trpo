
import os
import pprint
import tensorflow as tf

'''
Purpose: log output and statistics to STDOUT and TensorBoard
'''
class Auditor(object):

    def __init__(self, instance_name):

        current_dir = os.getcwd()
        log_dir = str(current_dir) + '/logs/' + instance_name

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_collection = {}
        self.log_filename = log_dir + '/' + 'log.txt'

        self.summary_writer = tf.summary.FileWriter(logdir=log_dir)

    ''' write to tensorboard '''
    def _write2tb(self):
        summary = tf.Summary()
        for key, value in self.log_collection.items():
            if isinstance(value, str) is False:
                if key == 'episode_number':
                    continue
                summary.value.add(tag='log/%s' % key, simple_value = value)

        self.summary_writer.add_summary(summary, self.log_collection['episode_number'])
        self.summary_writer.flush()

    def _write2file(self):
        with open(self.log_filename, "a") as fout:
            print('\n')
            pprint.pprint(self.log_collection, fout)

    def _print(self):
        print('\n')
        pprint.pprint(self.log_collection)

    def _clear(self):
        self.log_collection = {}

    def update(self, items):
        self.log_collection.update(items)

    def logmeta(self):
        self._print()
        self._write2file()
        self._clear()

    def log(self):
        self._print()
        self._write2file()
        self._write2tb()
        self._clear()

