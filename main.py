import getopt
import os
import platform
import socket
import sys
import time
from datetime import datetime

import numba
import numpy as np
import pandas as pd
import psutil
import pytz
from sklearn.metrics import accuracy_score

from multirocket.multirocket import MultiRocket
from utils.data_loader import read_univariate_ucr, non_109_datasets
from utils.tools import create_directory

pd.set_option('display.max_columns', 500)

data_path = "data/sample/"
problem = "ECG200"
itr = 0
kernel_selection = 0
num_features = 10000
feature_id = 202
save = True
num_threads = 0

try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:i:f:n:k:p:t:s:",
                               ["data_path=", "iter=", "featureid=", "num_features=",
                                "kernel_selection=", "problem=", "num_threads=", "save="])
except getopt.GetoptError:
    print("main.py -d <data_path> -i <iteration> -f <featureid> -n <num_features>"
          "-k <kernel_selection> -p <problem> -t <num_threads> -s <save>")
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print("main.py -d <data_path> -i <iteration> -f <featureid> -n <num_features>"
              "-k <kernel_selection> -p <problem> -t <num_threads> -s <save>")
        sys.exit()
    elif opt in ("-d", "--data_path"):
        data_path = arg
    elif opt in ("-i", "--iter"):
        itr = int(arg)
    elif opt in ("-f", "--featureid"):
        feature_id = int(arg)
    elif opt in ("-n", "--num_features"):
        num_features = int(arg)
    elif opt in ("-k", "--kernel_selection"):
        kernel_selection = int(arg)
    elif opt in ("-p", "--problem"):
        problem = arg
    elif opt in ("-t", "--num_threads"):
        num_threads = min(int(arg), psutil.cpu_count(logical=True))
    elif opt in ("-s", "--save"):
        save = bool(int(arg))

if __name__ == '__main__':
    data_folder = data_path + problem + "/"
    if os.path.exists(data_folder):
        if num_threads > 0:
            numba.set_num_threads(num_threads)
        output_path = os.getcwd() + "/output/"

        classifier_name = "MultiRocket{}_{}_{}".format(kernel_selection, feature_id, num_features)

        flag = False
        start = time.perf_counter()

        output_dir = "{}/multirocket/resample_{}/{}/{}/".format(output_path,
                                                                itr,
                                                                classifier_name,
                                                                problem)
        if save:
            create_directory(output_dir)

        print("=======================================================================")
        print("Starting Experiments")
        print("=======================================================================")
        print("Data path: {}".format(data_path))
        print("Output Dir: {}".format(output_dir))
        print("Iteration: {}".format(itr))
        print("Problem: {}".format(problem))
        print("Feature ID: {}".format(feature_id))
        print("Number of Features: {}".format(num_features))
        print("Kernels Selection: {}".format(kernel_selection))

        # set data folder
        data_folder = data_path + problem + "/"
        train_file = data_folder + problem + "_TRAIN.tsv"
        test_file = data_folder + problem + "_TEST.tsv"

        if kernel_selection == 0:
            print("Loading data")
            X_train, y_train = read_univariate_ucr(train_file, normalise=False)
            X_test, y_test = read_univariate_ucr(test_file, normalise=False)

            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
        else:
            print("Loading data")
            X_train, y_train = read_univariate_ucr(train_file, normalise=True)
            X_test, y_test = read_univariate_ucr(test_file, normalise=True)

        if (itr > 0) and (problem not in non_109_datasets):
            all_data = np.vstack((X_train, X_test))
            all_labels = np.hstack((y_train, y_test))
            print(all_data.shape)

            all_indices = np.arange(len(all_data))
            training_indices = np.loadtxt("data/indices109/{}_INDICES_TRAIN.txt".format(problem),
                                          skiprows=itr,
                                          max_rows=1).astype(np.int32)
            test_indices = np.setdiff1d(all_indices, training_indices, assume_unique=True)

            X_train, y_train = all_data[training_indices, :], all_labels[training_indices]
            X_test, y_test = all_data[test_indices, :], all_labels[test_indices]

        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

        classifier = MultiRocket(num_features=num_features,
                                 feature_id=feature_id,
                                 kernel_selection=kernel_selection)
        yhat_train = classifier.fit(X_train, y_train,
                                    predict_on_train=False)
        if yhat_train is not None:
            train_acc = accuracy_score(y_train, yhat_train)
        else:
            train_acc = -1

        yhat_test = classifier.predict(X_test)
        test_acc = accuracy_score(y_test, yhat_test)

        # get cpu information
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        max_freq = cpu_freq.max
        min_freq = cpu_freq.min
        memory = np.round(psutil.virtual_memory().total / 1e9)

        df_metrics = pd.DataFrame(data=np.zeros((1, 20), dtype=np.float), index=[0],
                                  columns=['timestamp', 'itr', 'classifier',
                                           'num_features', 'kernels_selection',
                                           'dataset',
                                           'train_acc', 'train_time',
                                           'test_acc', 'test_time',
                                           'generate_kernel_time',
                                           'apply_kernel_on_train_time',
                                           'apply_kernel_on_test_time',
                                           'machine', 'processor',
                                           'physical_cores',
                                           "logical_cores",
                                           'max_freq', 'min_freq', 'memory'])
        df_metrics["timestamp"] = datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
        df_metrics["itr"] = itr
        df_metrics["classifier"] = classifier_name
        df_metrics["num_features"] = num_features
        df_metrics["kernels_selection"] = kernel_selection
        df_metrics["dataset"] = problem
        df_metrics["train_acc"] = train_acc
        df_metrics["train_time"] = classifier.train_duration
        df_metrics["test_acc"] = test_acc
        df_metrics["test_time"] = classifier.test_duration
        df_metrics["generate_kernel_time"] = classifier.generate_kernel_duration
        df_metrics["apply_kernel_on_train_time"] = classifier.apply_kernel_on_train_duration
        df_metrics["apply_kernel_on_test_time"] = classifier.apply_kernel_on_test_duration
        df_metrics["machine"] = socket.gethostname()
        df_metrics["processor"] = platform.processor()
        df_metrics["physical_cores"] = physical_cores
        df_metrics["logical_cores"] = logical_cores
        df_metrics["max_freq"] = max_freq
        df_metrics["min_freq"] = min_freq
        df_metrics["memory"] = memory

        print(df_metrics)
        if save:
            df_metrics.to_csv(output_dir + 'results.csv', index=False)
