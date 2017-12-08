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
class TestParallelWorkflow(sl.WorkflowTask):
    def workflow(self):
        gen_step = self.new_task('gen', GenerateData)

        test_parallel = []
        for ii in range(10):
            test = self.new_task('test_{}'.format(ii), TestTask, number=ii)
            test.in_data= gen_step.out_data
            test_parallel.append(test)

        return test_parallel


# ------------------------------------------------------------------------
# Task classes
# ------------------------------------------------------------------------

class GenerateData(sl.ExternalTask):
    seed = luigi.IntParameter(348146)

    def out_data(self):
        return sl.TargetInfo(
            self, os.path.join(data_storage_path, "data_{}.pkl".format(self.seed))
        )

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
        return sl.TargetInfo(
            self, os.path.join(data_storage_path, "test_output_{}_{}.pkl".format(self.seed, self.number))
        )

    def run(self):
        with open(self.in_data().path, 'rb') as f:
            data_in = pickle.load(f)

        data_hash = hashlib.md5(data_in.tostring())

        # Just pause for a bit
        for ii in range(10):
            time.sleep(1)
            print("Task {} Iter {}".format(self.task_id, ii))

        # Save metadata:
        with self.out_task().open('w') as f:
            f.write("P: {}  Data hash: {}".format(self.number, data_hash))

if __name__ == '__main__':
    sl.run(main_task_cls=TestParallelWorkflow, cmdline_args=['--workers=4'])
