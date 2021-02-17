# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Effective summary statistics for convolutional outputs in time series classification
# https://arxiv.org/abs/2102.00457

import time

import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from multirocket import feature_names, get_feature_set
# from multirocket import minirocket as minirocket
from multirocket import minirocket_multivariate as minirocket  # use multivariate version.
from multirocket import rocket as rocket


class MultiRocket:

    def __init__(self,
                 num_features=10000,
                 feature_id=22,
                 kernel_selection=0):
        """
        MultiRocket
        :param num_features: number of features
        :param feature_id: feature id to identify the feature combinations
        :param kernel_selection: 0 = minirocket kernels, 1 = rocket kernels
        """
        if kernel_selection == 0:
            self.name = "MiniRocket_{}_{}".format(feature_id, num_features)
        else:
            self.name = "Rocket_{}_{}".format(feature_id, num_features)

        self.kernels = None
        self.kernel_selection = kernel_selection
        self.num_features = num_features
        self.feature_id = feature_id

        # get parameters based on feature id
        fixed_features, optional_features, num_random_features = get_feature_set(feature_id)
        self.fixed_features = fixed_features
        self.optional_features = optional_features
        self.num_random_features = num_random_features
        self.n_features_per_kernel = len(fixed_features) + num_random_features
        self.num_kernels = int(num_features / self.n_features_per_kernel)

        fixed_features_list = [feature_names[x] for x in self.fixed_features]
        optional_features_list = [feature_names[x] for x in self.optional_features]
        self.feature_list = fixed_features_list + optional_features_list
        print('FeatureID: {} -- features for each kernel: {}'.format(self.feature_id, self.feature_list))

        # if feature_id >= 100:
        #     self.n_features_per_kernel = int(feature_id / 100)
        # elif feature_id == 52:
        #     self.n_features_per_kernel = 22
        # else:
        #     tmp = feature_names[feature_id].split("+")
        #     self.n_features_per_kernel = len(tmp)

        print('Creating {} with {} kernels'.format(self.name, self.num_kernels))
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                                            normalize=True)

        self.train_duration = 0
        self.test_duration = 0
        self.generate_kernel_duration = 0
        self.apply_kernel_on_train_duration = 0
        self.apply_kernel_on_test_duration = 0

    def fit(self, x_train, y_train, predict_on_train=True):
        start_time = time.perf_counter()

        print('[{}] Training with training set of {}'.format(self.name, x_train.shape))
        if self.kernel_selection == 0:
            _start_time = time.perf_counter()
            self.kernels = minirocket.fit(x_train,
                                          self.feature_id,
                                          self.n_features_per_kernel,
                                          num_features=self.num_kernels)
            self.generate_kernel_duration = time.perf_counter() - _start_time

            x_train_transform = minirocket.transform(x_train, self.kernels)
            self.apply_kernel_on_train_duration = time.perf_counter() - _start_time - self.generate_kernel_duration
        else:
            _start_time = time.perf_counter()
            self.kernels = rocket.generate_kernels(x_train.shape[1],
                                                   num_kernels=self.num_kernels,
                                                   feature_id=self.feature_id,
                                                   num_features=self.n_features_per_kernel,
                                                   num_channels=x_train.shape[2])
            self.generate_kernel_duration = time.perf_counter() - _start_time

            x_train_transform = rocket.apply_kernels(x_train, self.kernels, self.feature_id)
            self.apply_kernel_on_train_duration = time.perf_counter() - start_time - self.generate_kernel_duration

        x_train_transform = np.nan_to_num(x_train_transform)

        elapsed_time = time.perf_counter() - start_time
        print('Kernels applied!, took {}s'.format(elapsed_time))
        print('Transformed Shape {}'.format(x_train_transform.shape))

        print('Training')
        _start_time = time.perf_counter()
        self.classifier.fit(x_train_transform, y_train)
        self.train_duration = time.perf_counter() - _start_time

        print('Training done!, took {:.3f}s'.format(self.train_duration))
        if predict_on_train:
            yhat = self.classifier.predict(x_train_transform)
        else:
            yhat = None

        return yhat

    def predict(self, x):
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()

        if self.kernel_selection == 0:
            x_test_transform = minirocket.transform(x, self.kernels)
        else:
            x_test_transform = rocket.apply_kernels(x, self.kernels, self.feature_id)

        self.apply_kernel_on_test_duration = time.perf_counter() - start_time
        x_test_transform = np.nan_to_num(x_test_transform)
        print('Kernels applied!, took {:.3f}s. Transformed shape: {}. '.format(self.apply_kernel_on_test_duration,
                                                                               x_test_transform.shape))

        yhat = self.classifier.predict(x_test_transform)
        self.test_duration = time.perf_counter() - start_time

        print("[{}] Predicting completed, took {:.3f}s".format(self.name, self.test_duration))

        return yhat
