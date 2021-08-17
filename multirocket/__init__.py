# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Effective summary statistics for convolutional outputs in time series classification
# https://arxiv.org/abs/2102.00457

import numpy as np
from numba import njit


from catch22A.catch22_full import sb_binarystats_mean_longstretch_1, sb_motif_three_quantile_hh, \
    co_histogram_ami_even_2_5, dn_histogram_mode_5, dn_histogram_mode_10, co_f1ecac, co_firstmin_ac, co_trev_1_num, \
    md_hrv_classic_pnn40, sb_transition_matrix_3ac_sum_diag_cov, pd_periodicity_wang_th0_01, \
    co_embed2_dist_tau_d_expfit_meandiff, in_auto_mutual_info_stats_40_gaussian_fmmi, fc_local_simple_mean1_tauresrat, \
    dn_outlier_include_p_001_mdrmd, dn_outlier_include_n_001_mdrmd, sp_summaries_welch_rect_area_5_1, \
    sb_binarystats_diff_longstretch_0, sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1, \
    sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1, sp_summaries_welch_rect_centroid, fc_local_simple_mean3_stderr
from utils.tools import numba_fft_v, autocorr

__version__ = "0.0.1"

def get_module_version():
    return __version__

# Original Rocket is feature 50 with Rocket Kernels
# Original MiniRocket is feature 23 with Minirocket Kernels
# single features
base_feature_names = [
    "DN_HistogramMode_5",  # 0
    "DN_HistogramMode_10",  # 1
    "CO_f1ecac",  # 2
    "CO_FirstMin_ac",  # 30
    "CO_HistogramAMI_even_2_5",  # 4
    "CO_trev_1_num",  # 5
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",  # 7
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",  # 10
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1",  # 15
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",  # 17
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",  # 20
    "FC_LocalSimple_mean3_stderr",
    "LongStretch_above_0",  # 22
    "PPV",  # 23
    "Max",
    "Mean",  # 25
]  # 26 features
ppv_paired_feature_names = ["PPV+" + x for x in base_feature_names]  # 26 features (52 features)
feature_names = base_feature_names + \
                ppv_paired_feature_names + \
                [
                    "All_Catch22",  # 52

                    "PPV+LongStretch1+Motif3",  # 53
                    "PPV+LongStretch1+HistogramAMI",
                    "PPV+Motif3+HistogramAMI",
                    "PPV+LongStretch1+Motif3+HistogramAMI",
                    "LongStretch1+Motif3+HistogramAMI",  # 57

                    "PPV+LongStretchAbove0+Motif3",  # 58
                    "PPV+LongStretchAbove0+HistogramAMI",
                    "PPV+LongStretchAbove0+Motif3+HistogramAMI",
                    "LongStretchAbove0+Motif3+HistogramAMI",  # 61

                    "LongStretchAbove0+Max",  # 62
                    "LongStretchAbove0+MeanLongStretch1+Motif3+HistogramAMI",  # 63
                    "PPV+LongStretchAbove0+MeanLongStretch1+Motif3+HistogramAMI",  # 64
                    "PPV+LongStretchAbove0+MeanLongStretch1+Motif3+HistogramAMI+Max"  # 65
                ]


@njit(fastmath=True, cache=True)
def features_to_base_features(feature_id):
    """
    Factorise the features into base features
    :param feature_id: feature id
    :return: arrays of base feature ids
    """
    if (feature_id >= 26) and (feature_id < 52):
        # for all PPV-paired features, feature ids between 27 and 54
        output = np.zeros(2, dtype=np.int64)
        output[0] = 23
        output[1] = feature_id - 26
    elif feature_id == 52:
        # catch22 features
        output = np.arange(0, 22, dtype=np.int64)
    elif feature_id == 53:
        # PPV+LongStretch1+Motif3
        output = np.zeros(3, dtype=np.int64)
        output[0] = 23  # PPV
        output[1] = 7  # LongStretch1
        output[2] = 17  # Motif3
    elif feature_id == 54:
        # PPV+LongStretch1+HistogramAMI
        output = np.zeros(3, dtype=np.int64)
        output[0] = 23  # PPV
        output[1] = 4  # HistogramAMI
        output[2] = 7  # LongStretch1
    elif feature_id == 55:
        # PPV+Motif3+HistogramAMI
        output = np.zeros(3, dtype=np.int64)
        output[0] = 23  # PPV
        output[1] = 4  # HistogramAMI
        output[2] = 7  # Motif3
    elif feature_id == 56:
        # PPV+LongStretch1+Motif3+HistogramAMI
        output = np.zeros(4, dtype=np.int64)
        output[0] = 23  # PPV
        output[1] = 4  # HistogramAMI
        output[2] = 7  # LongStretch1
        output[3] = 17  # Motif3
    elif feature_id == 57:
        # LongStretch1+Motif3+HistogramAMI
        output = np.zeros(3, dtype=np.int64)
        output[0] = 4  # HistogramAMI
        output[1] = 7  # LongStretch1
        output[2] = 17  # Motif3
    elif feature_id == 58:
        # PPV+LongStretchAbove0+Motif3
        output = np.zeros(3, dtype=np.int64)
        output[0] = 23  # PPV
        output[1] = 22  # LongStretchAbove0
        output[2] = 17  # Motif3
    elif feature_id == 59:
        # PPV+LongStretchAbove0+HistogramAMI
        output = np.zeros(3, dtype=np.int64)
        output[0] = 23  # PPV
        output[1] = 22  # LongStretchAbove0
        output[2] = 4  # HistogramAMI
    elif feature_id == 60:
        # PPV+LongStretchAbove0+Motif3+HistogramAMI
        output = np.zeros(4, dtype=np.int64)
        output[0] = 23  # PPV
        output[1] = 22  # LongStretchAbove0
        output[2] = 4  # HistogramAMI
        output[3] = 17  # Motif3
    elif feature_id == 61:
        # LongStretchAbove0+Motif3+HistogramAMI
        output = np.zeros(3, dtype=np.int64)
        output[0] = 22  # LongStretchAbove0
        output[1] = 4  # HistogramAMI
        output[2] = 17  # Motif3
    elif feature_id == 62:
        # LongStretchAbove0+Max
        output = np.zeros(2, dtype=np.int64)
        output[0] = 22  # LongStretchAbove0
        output[1] = 24  # Max
    elif feature_id == 63:
        # LongStretchAbove0+MeanLongStretch1+Motif3+HistogramAMI
        output = np.zeros(4, dtype=np.int64)
        output[0] = 22  # LongStretchAbove0
        output[1] = 4  # HistogramAMI
        output[2] = 7  # LongStretch1
        output[3] = 17  # Motif3
    elif feature_id == 64:
        # "PPV+LongStretchAbove0+MeanLongStretch1+Motif3+HistogramAMI"  # 66
        output = np.zeros(5, dtype=np.int64)
        output[0] = 23  # PPV
        output[1] = 22  # LongStretchAbove0
        output[2] = 4  # HistogramAMI
        output[3] = 7  # MeanLongStretch1
        output[4] = 17  # Motif3
    elif feature_id == 65:
        # "PPV+LongStretchAbove0+MeanLongStretch1+Motif3+HistogramAMI+Max"  # 67
        output = np.zeros(6, dtype=np.int64)
        output[0] = 23  # PPV
        output[1] = 22  # LongStretchAbove0
        output[2] = 4  # HistogramAMI
        output[3] = 7  # MeanLongStretch1
        output[4] = 17  # Motif3
        output[5] = 24  # Max
    else:
        output = np.zeros(1, dtype=np.int64)
        output[0] = feature_id
    return output


@njit(fastmath=True, cache=True)
def get_base_features(y, y_min, y_max, y_mean, ppv, output_length, feature_id):
    """
    Get all the 26 base features
    """
    if feature_id == 0:
        return dn_histogram_mode_5(y, y_min, y_max)
    elif feature_id == 1:
        return dn_histogram_mode_10(y, y_min, y_max)
    elif feature_id == 2:
        y_mean = y_mean / output_length
        fft = numba_fft_v(y - y_mean)
        ac = autocorr(y, fft)
        return co_f1ecac(ac)
    elif feature_id == 3:
        y_mean = y_mean / output_length
        fft = numba_fft_v(y - y_mean)
        ac = autocorr(y, fft)
        return co_firstmin_ac(ac)
    elif feature_id == 4:
        return co_histogram_ami_even_2_5(y, y_min, y_max)
    elif feature_id == 5:
        return co_trev_1_num(y)
    elif feature_id == 6:
        return md_hrv_classic_pnn40(y)
    elif feature_id == 7:
        y_mean = y_mean / output_length
        return sb_binarystats_mean_longstretch_1(y, y_mean)
    elif feature_id == 8:
        y_mean = y_mean / output_length
        fft = numba_fft_v(y - y_mean)
        ac = autocorr(y, fft)
        return sb_transition_matrix_3ac_sum_diag_cov(y, ac)
    elif feature_id == 9:
        return pd_periodicity_wang_th0_01(y)
    elif feature_id == 10:
        y_mean = y_mean / output_length
        fft = numba_fft_v(y - y_mean)
        ac = autocorr(y, fft)
        return co_embed2_dist_tau_d_expfit_meandiff(y, ac)
    elif feature_id == 11:
        y_mean = y_mean / output_length
        fft = numba_fft_v(y - y_mean)
        ac = autocorr(y, fft)
        return in_auto_mutual_info_stats_40_gaussian_fmmi(ac)
    elif feature_id == 12:
        y_mean = y_mean / output_length
        fft = numba_fft_v(y - y_mean)
        ac = autocorr(y, fft)
        return fc_local_simple_mean1_tauresrat(y, ac)
    elif feature_id == 13:
        return dn_outlier_include_p_001_mdrmd(y)
    elif feature_id == 14:
        return dn_outlier_include_n_001_mdrmd(y)
    elif feature_id == 15:
        y_mean = y_mean / output_length
        fft = numba_fft_v(y - y_mean)
        return sp_summaries_welch_rect_area_5_1(y, fft)
    elif feature_id == 16:
        return sb_binarystats_diff_longstretch_0(y)
    elif feature_id == 17:
        return sb_motif_three_quantile_hh(y)
    elif feature_id == 18:
        return sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1(y)
    elif feature_id == 19:
        return sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1(y)
    elif feature_id == 20:
        y_mean = y_mean / output_length
        fft = numba_fft_v(y - y_mean)
        return sp_summaries_welch_rect_centroid(y, fft)
    elif feature_id == 21:
        return fc_local_simple_mean3_stderr(y)
    elif feature_id == 22:
        return sb_binarystats_mean_longstretch_1(y, 0)
    elif feature_id == 23:
        return ppv / output_length
    elif feature_id == 24:
        return y_max
    elif feature_id == 25:
        return y_mean / output_length

    # wrong feature_id
    return 0.0


@njit(fastmath=True, cache=True)
def get_feature_set(feature_id):
    """
    Get feature set based on feature id
    :param feature_id: feature id
    :return: F (set of fixed features), O (set of optional features), r (number of optional features
    """
    if feature_id == 101:
        # HistogramAMI, Motif3, LongStretch1, LongStretchAbove0, PPV, Max
        F = np.empty(0, dtype=np.int64)
        O = np.array((4, 7, 17, 22, 23, 24))
        r = 1
    elif feature_id == 201:
        # HistogramAMI, Motif3, LongStretch1, LongStretchAbove0, PPV, Max
        F = np.empty(0, dtype=np.int64)
        O = np.array((4, 7, 17, 22, 23, 24))
        r = 2
    elif feature_id == 202:
        # PPV paired with HistogramAMI, Motif3, LongStretch1, LongStretchAbove0, Max
        F = np.array((23,))
        O = np.array((4, 7, 17, 22, 24))
        r = 1
    elif feature_id == 203:
        # LongStretchAbove0 paired with HistogramAMI, Motif3, LongStretch1, PPV, Max
        F = np.array((22,))
        O = np.array((4, 7, 17, 23, 24))
        r = 1
    elif feature_id == 205:
        # PPV paired with HistogramAMI, Motif3, LongStretch1, Max
        F = np.array((23,))
        O = np.array((4, 7, 17, 24))
        r = 1
    elif feature_id == 301:
        # HistogramAMI, Motif3, LongStretch1, LongStretchAbove0, PPV, Max
        F = np.empty(0, dtype=np.int64)
        O = np.array((4, 7, 17, 22, 23, 24))
        r = 1
    elif feature_id == 302:
        # PPV paired with HistogramAMI, Motif3, LongStretch1, LongStretchAbove0, Max
        F = np.array((23,))
        O = np.array((4, 7, 17, 22, 24))
        r = 2
    elif feature_id == 303:
        # LongStretchAbove0 paired with HistogramAMI, Motif3, LongStretch1, PPV, Max
        F = np.array((22,))
        O = np.array((4, 7, 17, 23, 24))
        r = 2
    elif feature_id == 304:
        # PPV+LongStretchAbove0 paired with HistogramAMI, Motif3, LongStretch1, Max
        F = np.array((22, 23))
        O = np.array((4, 7, 17, 24))
        r = 1
    elif feature_id == 305:
        # PPV paired with HistogramAMI, Motif3, LongStretch1, Max
        F = np.array((23,))
        O = np.array((4, 7, 17, 24))
        r = 2
    elif feature_id >= 100:
        # not supported yet, so return all base features
        F = np.empty(0, dtype=np.int64)
        O = np.arange(0, 26)
        r = 1
    else:
        F = features_to_base_features(feature_id)
        O = np.empty(0, dtype=np.int64)
        r = 0
    return F, O, r


@njit(fastmath=True, cache=True)
def sample_base_feature_ids(feature_selection, num_features, num_rows):
    """
    Sample feature ids
    :param feature_selection: a general ID to select the features
    :param num_features:   number of features per kernel
    :param num_rows: either number of kernels (for random) or number of estimators
    :return: array of base feature ids with shape (num_rows x num_features)
    """
    fixed_features, optional_features, r = get_feature_set(feature_selection)
    # feature_ids shape num_rows
    if feature_selection == 304:
        # set the first 2 features as PPV and LongStretchAbove0
        # uniformly sample the rest for the remaining rows
        feature_ids = np.full((num_features, num_rows), fixed_features[0], dtype=np.int32)
        feature_ids[1] = fixed_features[1]
        feature_ids[2] = np.random.choice(optional_features, num_rows)
        feature_ids = feature_ids.T
    elif feature_selection in (202, 204, 205, 302, 303, 305):
        # set the first feature as PPV
        # uniformly sample the rest for the remaining rows
        feature_ids = np.full((num_rows, num_features),
                              fixed_features[0], dtype=np.int32)
        for i in range(num_rows):
            feature_ids[i, 1:] = np.random.choice(optional_features, size=r, replace=False)

    elif feature_selection >= 100:
        # uniformly sample the features from the pool
        feature_ids = np.empty((num_rows, num_features), dtype=np.int32)
        for i in range(num_rows):
            feature_ids[i] = np.random.choice(optional_features, size=r, replace=False)
    else:
        # just assign the features same feature to all kernels
        feature_ids = np.empty((num_rows, len(fixed_features)), dtype=np.int32)
        for i in range(num_rows):
            feature_ids[i] = fixed_features
    return feature_ids
