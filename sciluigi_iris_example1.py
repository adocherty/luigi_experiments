import csv
import logging

import luigi
import sciluigi as sl
import os
import numpy as np
import pickle

from sklearn import datasets, svm, model_selection

# Hmmm?
data_storage_path = "./output"

# Example paramters
splitter_params = {'seed': 1234, 'train_size': 0.6}
input_params = {'filename': '/opt/data/iris.csv'}
clf_params = {'seed': 1234, 'C': 1.0, 'C_scan': [1, 5, 10]}

# ------------------------------------------------------------------------
# Init logging
# ------------------------------------------------------------------------

log = logging.getLogger('sciluigi-interface')


# ------------------------------------------------------------------------
# Workflow class
# ------------------------------------------------------------------------
class TrainWorkflow(sl.WorkflowTask):
    def workflow(self):
        input = self.new_task('input', InputData)
        splitter = self.new_task('splitter', Splitter)
        train = self.new_task('train', TrainClassifier)

        splitter.in_data = input.out_data
        train.in_train = splitter.out_split0

        return train


class TrainScan(sl.WorkflowTask):
    def workflow(self):
        input = self.new_task('input', InputData)

        splitter = self.new_task('splitter', Splitter)
        splitter.in_data = input.out_data

        train_scan = []
        for C in clf_params['C_scan']:
            train = self.new_task('train', TrainClassifier, svm_C=C)
            train.in_train = splitter.out_split0
            train_scan.append(train)

        return train_scan


# ------------------------------------------------------------------------
# Task classes
# ------------------------------------------------------------------------
class InputData(sl.ExternalTask):
    filename = luigi.Parameter(input_params['filename'])

    def out_data(self):
        return sl.TargetInfo(self, self.filename)


class Splitter(sl.Task):
    seed = luigi.IntParameter(default=splitter_params['seed'])
    train_size = luigi.IntParameter(default=splitter_params['train_size'])

    in_data = None

    def out_task(self):
        return sl.TargetInfo(self, 'output_{}.txt'.format(self.task_id))

    def out_split0(self):
        return sl.TargetInfo(self, os.path.join(data_storage_path, "data_split_0.pkl"))

    def out_split1(self):
        return sl.TargetInfo(self, os.path.join(data_storage_path, "data_split_1.pkl"))

    def run(self):
        # Read data:
        with self.in_data().open('r') as csv_file:
            data_file = csv.reader(csv_file)
            temp = next(data_file)
            n_samples = int(temp[0])
            n_features = int(temp[1])
            data_in = np.empty((n_samples, n_features + 1), dtype=np.float32)
            for i, ir in enumerate(data_file):
                data_in[i] = np.asarray(ir, dtype=np.float32)

        # Random class: all randomness should use this
        rnd = np.random.RandomState(self.seed)

        # Split using sklearn ShuffleSplit
        ss = model_selection.ShuffleSplit(n_splits=1, train_size=self.train_size, random_state=rnd)
        train_index, test_index = next(ss.split(data_in))

        # Note: luigi Targets only support text outputs,
        # not binary outputs with the open method
        with open(self.out_split0().path, 'wb') as f:
            pickle.dump(data_in[train_index], f)

        with open(self.out_split1().path, 'wb') as f:
            pickle.dump(data_in[test_index], f)

        # Save metadata:
        with self.out_task().open('w') as f:
            f.write("OK - train {}\n test{}".format(train_index, test_index))


class TrainClassifier(sl.Task):
    seed = luigi.IntParameter(default=clf_params['seed'])
    svm_C = luigi.IntParameter(default=clf_params['C'])

    in_train = None

    def out_task(self):
        return sl.TargetInfo(self, 'output_{}.txt'.format(self.task_id))

    # def out_model(self):
    #     return sl.TargetInfo(self, 'model_{}.txt'.format(self.task_id))

    def run(self):
        with open(self.in_train().path, 'rb') as f:
            data_in = pickle.load(f)

        classifier = svm.SVC(C=self.svm_C)
        X = data_in[:, :-1]
        Y = data_in[:, -1]
        classifier.fit(X, Y)

        print("Train accuracy:", classifier.score(X, Y))

        # We could pickle the model here


        # Save metadata:
        with self.out_task().open('w') as f:
            f.write("OK")

if __name__ == '__main__':
    sl.run_local(main_task_cls=TrainScan)