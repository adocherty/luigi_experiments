import logging
import time

import six

import luigi
import sciluigi as sl
import os
import numpy as np
import pickle
import warnings

from sklearn import datasets, svm, model_selection
import hashlib

from luigi import configuration

# Hmmm?
data_storage_path = "./output"

# ------------------------------------------------------------------------
# Init logging
# ------------------------------------------------------------------------
log = logging.getLogger('sciluigi-interface')

# ------------------------------------------------------------------------
# Workflow class
# ------------------------------------------------------------------------
class TestCachedTarget2(sl.WorkflowTask):
    def workflow(self):
        gen_step = self.new_task('gen', GenerateData)

        test_parallel = []
        for ii in range(4):
            test = self.new_task('test_{}'.format(ii), TestTask, number=ii)
            test.in_data= gen_step.out_data
            test_parallel.append(test)

        return test_parallel


class TestCachedTarget(sl.WorkflowTask):
    def workflow(self):
        gen_step = self.new_task('gen', GenerateData)

        test = self.new_task('test', TestTask, number=4)
        test.in_data = gen_step.out_data
        return test


# ------------------------------------------------------------------------
# A target info object better suited to us?
# ------------------------------------------------------------------------

class CachedTarget(sl.TargetInfo):
    # Don't use the values of the SciLuigi task parameters
    _unhashable_parameters = ['workflow_task', 'instance_name']

    def __init__(self, task, base_path=None, suffix=".data", format=None, is_tmp=False):
        self.task = task
        self.basepath = base_path
        self.suffix = suffix

    @property
    def path(self):
        name = self.task.__class__.__name__
        h = self.get_hash()
        #print("Task {}, hash {}".format(self.task.__class__.__name__, h))
        return os.path.join(self.basepath, "{}_{}{}".format(name, h, self.suffix))

    @property
    def target(self):
        return luigi.LocalTarget(self.path)

    def get_hash(self, h=None):
        # Get input classes
        input_list = [
            (attr, self.task.__dict__[attr]())
            for attr in self.task.__dict__  if 'in_' == attr[0:3]
        ]

        # At the top level, generate the hashing object
        if h is None:
            h = hashlib.md5()

        # Get input hashes
        for in_name, input in input_list:
            if hasattr(input, "get_hash"):
                input.get_hash(h)
            else:
                warnings.warn("CachedTarget used with non-cached targets as inputs.")

        # Create output hash
        params = [
            p for p in self.task.get_param_names() if p not in self._unhashable_parameters
            ]

        for p in params:
            h.update("{} = {}".format(p, self.task.__dict__[p]).encode('utf8'))

        return h.hexdigest()


# ------------------------------------------------------------------------
# Task classes
# ------------------------------------------------------------------------

class GenerateData(sl.ExternalTask):
    seed = luigi.IntParameter(348146)

    def out_data(self):
        return CachedTarget(self, data_storage_path)

    def run(self):
        # Random class: all randomness should use this
        rnd = np.random.RandomState(self.seed)

        # Write data to file
        data = rnd.randint(-10,10,(100,4))
        with open(self.out_data().path, 'wb') as f:
            pickle.dump(data, f)


class TestTask(sl.Task):
    seed = luigi.IntParameter(default=312423)
    number = luigi.IntParameter(default=0)

    in_data = None

    def out_task(self):
        return CachedTarget(self, data_storage_path)

    def run(self):
        with open(self.in_data().path, 'rb') as f:
            data_in = pickle.load(f)

        # Just pause for a bit
        for ii in range(20):
            time.sleep(1)
            print("Task {} Iter {}".format(self.task_id, ii))

        # Save metadata:
        with self.out_task().open('w') as f:
            f.write("P: {}  Data hash: {}".format(self.number, self.out_task().get_hash()))

if __name__ == '__main__':
    sl.run_local(main_task_cls=TestCachedTarget)
