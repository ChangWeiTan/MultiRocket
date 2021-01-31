# Angus Dempster, Francois Petitjean, Geoff Webb
#
# @article{dempster_etal_2020,
#   author  = {Dempster, Angus and Petitjean, Fran\c{c}ois and Webb, Geoffrey I},
#   title   = {ROCKET: Exceptionally fast and accurate time classification using random convolutional kernels},
#   year    = {2020},
#   journal = {Data Mining and Knowledge Discovery},
#   doi     = {https://doi.org/10.1007/s10618-020-00701-z}
# }
#
# https://arxiv.org/abs/1910.13051 (preprint)

import numpy as np
from numba import prange, njit

from multirocket import get_base_features, sample_base_feature_ids


# =======================================================================================================
# Rocket functions
# =======================================================================================================
@njit(fastmath=True, cache=True)
def generate_kernels(input_length, num_kernels, feature_id, num_features=2, num_channels=1):
    """
    Generate kernels from a subset of features performing better than MAX
    """
    feature_ids = sample_base_feature_ids(feature_id, num_features, num_kernels)

    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    candidate_lengths = candidate_lengths[candidate_lengths < input_length]
    lengths = np.random.choice(candidate_lengths, num_kernels)

    num_channel_indices = (2 ** np.random.uniform(0, np.log2(num_channels + 1), num_kernels)).astype(np.int32)
    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros((num_channels, lengths.sum()), dtype=np.float64)
    if feature_id != 50:
        biases = np.zeros(num_kernels, dtype=np.float64)
    else:
        biases = np.zeros(num_kernels * 2, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    for i in range(num_kernels):
        _length = lengths[i]

        _weights = np.random.normal(0, 1, (num_channels, lengths[i]))

        a = lengths[:i].sum()
        b = a + lengths[i]
        for j in range(num_channels):
            _weights[j] = _weights[j] - _weights[j].mean()
        weights[:, a:b] = _weights

        a1 = num_channel_indices[:i].sum()
        b1 = a1 + num_channel_indices[i]
        channel_indices[a1:b1] = np.random.choice(np.arange(0, num_channels), num_channel_indices[i],
                                                  replace=False)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) // (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

    for i in range(len(biases)):
        biases[i] = np.random.uniform(-1, 1)

    return feature_ids, weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices


@njit(fastmath=True, cache=True)
def apply_kernel(X, feature_ids, weights, length, bias, dilation, padding,
                 num_channel_indices, channel_indices, stride):
    # zero padding
    if padding > 0:
        _input_length, _num_channels = X.shape
        _X = np.zeros((_input_length + (2 * padding), _num_channels))
        _X[padding:(padding + _input_length), :] = X
        X = _X

    input_length, num_channels = X.shape

    output_length = input_length - ((length - 1) * dilation)

    y = np.zeros(output_length, dtype=np.float64)
    y_min = np.PINF
    y_max = np.NINF
    y_mean = 0.0
    ppv = 0.0
    pos_mean = 0.0

    for i in range(0, output_length, stride):
        _sum = bias

        for j in range(length):
            for k in range(num_channel_indices):
                _sum += weights[channel_indices[k], j] * X[i + (j * dilation), channel_indices[k]]
        y[i] = _sum
        y_mean += _sum

        if _sum < y_min:
            y_min = _sum
        if _sum > y_max:
            y_max = _sum
        if _sum > 0:
            ppv += 1
            pos_mean += _sum

    outputs = np.empty((len(feature_ids)), dtype=np.float64)

    for i in range(len(feature_ids)):
        feature_id = feature_ids[i]
        outputs[i] = get_base_features(y, y_min, y_max, y_mean, ppv, output_length, feature_id)

    return outputs


@njit(fastmath=True, cache=True)
def apply_kernel_2_ppv(X, weights, length, bias1, bias2, dilation, padding,
                       num_channel_indices, channel_indices, stride):
    # zero padding
    if padding > 0:
        _input_length, _num_channels = X.shape
        _X = np.zeros((_input_length + (2 * padding), _num_channels))
        _X[padding:(padding + _input_length), :] = X
        X = _X

    input_length, num_channels = X.shape

    output_length = input_length - ((length - 1) * dilation)

    ppv1 = 0.0
    ppv2 = 0.0

    for i in range(0, output_length, stride):
        _sum = 0

        for j in range(length):
            for k in range(num_channel_indices):
                _sum += weights[channel_indices[k], j] * X[i + (j * dilation), channel_indices[k]]

        if _sum > bias1:
            ppv1 += 1
        if _sum > bias2:
            ppv2 += 1

    return ppv1 / output_length, ppv2 / output_length


@njit(parallel=True, fastmath=True, cache=True)
def apply_kernels(X, kernels, feature_id, stride=1):
    feature_ids, weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices = kernels

    num_kernels, n_features = feature_ids.shape
    num_examples, _, _ = X.shape

    _X = np.zeros((num_examples, num_kernels * n_features), dtype=np.float64)

    for i in prange(num_examples):

        a1 = 0  # for weights
        a2 = 0  # for features
        a3 = 0  # for channels

        for j in range(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + n_features
            b3 = a3 + num_channel_indices[j]
            if feature_id != 49:
                _X[i, a2:b2] = apply_kernel(X[i], feature_ids[j],
                                            weights[:, a1:b1], lengths[j], biases[j], dilations[j], paddings[j],
                                            num_channel_indices[j], channel_indices[a3:b3], stride)
            else:
                _X[i, a2:b2] = apply_kernel_2_ppv(X[i],
                                                  weights[:, a1:b1], lengths[j], biases[j], biases[j + num_kernels],
                                                  dilations[j], paddings[j],
                                                  num_channel_indices[j], channel_indices[a3:b3], stride)

            a1 = b1
            a2 = b2
            a3 = b3

    return _X
