# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Effective summary statistics for convolutional outputs in time series classification
# https://arxiv.org/abs/2102.00457

import numpy as np
from numba import njit

# =======================================================================================================
# Catch22 features
# Implemented with Numba, following the Java implementation from
# https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/tsml/transformers/Catch22.java
# =======================================================================================================
from catch22A.catch22_basis import histogram_mode, outlier_include, summaries_welth_rect, local_simple_mean, \
    co_first_zero, fluct_prop, splinefit
from utils.tools import autocorr, numba_std, numba_fft_v


@njit(fastmath=True, cache=True)
def compute_features(i, y, y_max, y_min, y_mean, fft, ac):
    if i == 0:
        return dn_histogram_mode_5(y, y_min, y_max)
    elif i == 1:
        return dn_histogram_mode_10(y, y_min, y_max)
    elif i == 2:
        return co_f1ecac(ac)
    elif i == 3:
        return co_firstmin_ac(ac)
    elif i == 4:
        return co_histogram_ami_even_2_5(y, y_min, y_max)
    elif i == 5:
        return co_trev_1_num(y)
    elif i == 6:
        return md_hrv_classic_pnn40(y)
    elif i == 7:
        return sb_binarystats_mean_longstretch_1(y, y_mean)
    elif i == 8:
        return sb_transition_matrix_3ac_sum_diag_cov(y, ac)
    elif i == 9:
        return pd_periodicity_wang_th0_01(y)
    elif i == 10:
        return co_embed2_dist_tau_d_expfit_meandiff(y, ac)
    elif i == 11:
        return in_auto_mutual_info_stats_40_gaussian_fmmi(ac)
    elif i == 12:
        return fc_local_simple_mean1_tauresrat(y, ac)
    elif i == 13:
        return dn_outlier_include_p_001_mdrmd(y)
    elif i == 14:
        return dn_outlier_include_n_001_mdrmd(y)
    elif i == 15:
        return sp_summaries_welch_rect_area_5_1(y, fft)
    elif i == 16:
        return sb_binarystats_diff_longstretch_0(y)
    elif i == 17:
        return sb_motif_three_quantile_hh(y)
    elif i == 18:
        return sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1(y)
    elif i == 19:
        return sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1(y)
    elif i == 20:
        return sp_summaries_welch_rect_centroid(y, fft)
    elif i == 21:
        return fc_local_simple_mean3_stderr(y)
    return 0


@njit(fastmath=True, cache=True)
def catch22_full(y, y_max, y_min, y_mean, fft, ac, feature_id=22):
    """
    Compute all Catch22 features
    out(0) = DN_HistogramMode_5;
    out(1) = DN_HistogramMode_10
    out(2) = CO_f1ecac
    out(3) = CO_FirstMin_ac
    out(4) = CO_HistogramAMI_even_2_5
    out(5) = CO_trev_1_num
    out(6) = MD_hrv_classic_pnn40
    out(7) = SB_BinaryStats_mean_longstretch1
    out(8) = SB_TransitionMatrix_3ac_sumdiagcov
    out(9) = PD_PeriodicityWang_th0_01
    out(10) = CO_Embed2_Dist_tau_d_expfit_meandiff
    out(11) = IN_AutoMutualInfoStats_40_gaussian_fmmi
    out(12) = FC_LocalSimple_mean1_tauresrat
    out(13) = DN_OutlierInclude_p_001_mdrmd
    out(14) = DN_OutlierInclude_n_001_mdrmd
    out(15) = SP_Summaries_welch_rect_area_5_1
    out(16) = SB_BinaryStats_diff_longstretch0
    out(17) = SB_MotifThree_quantile_hh
    out(18) = SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1
    out(19) = SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1
    out(20) = SP_Summaries_welch_rect_centroid
    out(21) = FC_LocalSimple_mean3_stderr

    :param y: time series
    :param y_max: max of the time series
    :param y_min: min of the time series
    :param y_mean: mean of the time series
    :param fft: FFT of the time series
    :param ac: auto-correlation vector of the time series
    :return: 22 features
    """
    if feature_id == 22:
        out = np.zeros(22, dtype=np.float64)
        for i in range(22):
            out[i] = compute_features(i, y, y_max, y_min, y_mean, fft, ac)
    else:
        out = np.zeros(1, dtype=np.float64)
        out[0] = compute_features(feature_id, y, y_max, y_min, y_mean, fft, ac)
    return out


@njit(fastmath=True, cache=True)
def compute_catch22(y):
    y_max = np.max(y)
    y_min = np.min(y)
    y_mean = np.mean(y)
    fft = numba_fft_v(y - y_mean)
    ac = autocorr(y, fft)

    return catch22_full(y, y_max, y_min, y_mean, fft, ac)


# ==================================================================
# Distribution
# ==================================================================
@njit(fastmath=True, cache=True)
def dn_histogram_mode_5(y, y_min, y_max):
    """
    Mode of z-scored distribution (5-bin histogram)
    """
    return histogram_mode(y, y_min, y_max, 5)


@njit(fastmath=True, cache=True)
def dn_histogram_mode_10(y, y_min, y_max):
    """
    Mode of z-scored distribution (10-bin histogram)
    """
    return histogram_mode(y, y_min, y_max, 10)


# ==================================================================
# Simple Temporal Statistics
# ==================================================================
@njit(fastmath=True, cache=True)
def sb_binarystats_mean_longstretch_1(y, y_mean):
    """
    Longest period of consecutive values above the mean
    """
    len_y = len(y)
    last_val = 0
    max_stretch = 0.0
    for i in range(len_y):
        if ((y[i] - y_mean) < 0) or (i == (len_y - 1)):
            stretch = i - last_val
            if stretch > max_stretch:
                max_stretch = stretch
            last_val = i

    return np.float64(max_stretch)


@njit(fastmath=True, cache=True)
def dn_outlier_include_p_001_mdrmd(y):
    """
    Time intervals between successive extreme events above the mean
    """
    return outlier_include(y)


@njit(fastmath=True, cache=True)
def dn_outlier_include_n_001_mdrmd(y):
    """
    Time intervals between successive extreme events below the mean
    """
    return outlier_include(-y)


# ==================================================================
# Linear autocorrelation
# ==================================================================
@njit(fastmath=True, cache=True)
def co_f1ecac(ac):
    """
    First 1/e crossing of autocorrelation function
    """
    series_len = len(ac)  # time-series length
    # thresh = 0.36787944117144233  # 1/e threshold
    thresh = 1 / np.exp(1)  # 1/e threshold

    for i in range(1, series_len):
        if (ac[i - 1] - thresh) * (ac[i] - thresh) < 0:  # cross the 1/e line
            return np.float64(i)

    return np.float64(series_len)


@njit(fastmath=True, cache=True)
def co_firstmin_ac(ac):
    """
    First minimum of autocorrelation function
    """
    series_len = len(ac)  # time series length

    # factorising the matlab code as there was repeating ac computation
    for i in range(1, series_len - 1):
        if (ac[i] < ac[i - 1]) and (ac[i] < ac[i + 1]):
            return np.float64(i)

    return np.float64(series_len)


@njit(fastmath=True, cache=True)
def sp_summaries_welch_rect_area_5_1(y, fft):
    """
    Total power in lowest fifth of frequencies in the Fourier power spectrum
    """
    return summaries_welth_rect(y, False, fft)


@njit(fastmath=True, cache=True)
def sp_summaries_welch_rect_centroid(y, fft):
    """
    Centroid of the Fourier power spectrum
    """
    return summaries_welth_rect(y, True, fft)


@njit(fastmath=True, cache=True)
def fc_local_simple_mean3_stderr(y):
    """
    Mean error from a rolling 3-sample mean forecasting
    """
    if len(y) < 6:
        return 0.0
    res, mean_res = local_simple_mean(y, 3)
    return numba_std(res, mean_res)


# ==================================================================
# Nonlinear autocorrelation
# ==================================================================
@njit(fastmath=True, cache=True)
def co_trev_1_num(y):
    """
    Time-reversibility statistic, ((x_t+1 − x_t)^3)_t
    """
    a = len(y) - 1
    if a == 0:
        return 0.0
    mean = 0
    for i in range(a):
        mean += (y[i + 1] - y[i]) ** 3
    mean /= a
    return mean


@njit(fastmath=True, cache=True)
def co_histogram_ami_even_2_5(y, y_min, y_max):
    """
    Automutual information, m = 2, τ = 5
    """
    len_y = len(y)
    if len_y <= 2:
        return 0.0

    new_min = y_min - 0.1
    new_max = y_max + 0.1
    bin_width = (new_max - new_min) / 5

    histogram = np.zeros((5, 5), dtype=np.float64)
    sumx = np.zeros(5, dtype=np.float64)
    sumy = np.zeros(5, dtype=np.float64)

    v = 1 / (len_y - 2)
    for i in range(len_y - 2):
        idx1 = int((y[i] - new_min) / bin_width)
        idx2 = int((y[i + 2] - new_min) / bin_width)

        histogram[idx1, idx2] += v
        sumx[idx1] += v
        sumy[idx2] += v

    ami = 0.0
    for i in range(5):
        for j in range(5):
            if histogram[i, j] > 0:
                ami += histogram[i, j] * np.log(histogram[i, j] / sumx[i] / sumy[j])
    return ami


@njit(fastmath=True, cache=True)
def in_auto_mutual_info_stats_40_gaussian_fmmi(ac):
    """
    First minimum of the automutual information function
    """
    tau = int(np.ceil(len(ac) / 2))
    tau = tau if tau < 40 else 40
    # tau = numba_min(40, int(np.ceil(len(ac) / 2)))

    diffs = np.zeros(tau - 1, dtype=np.float64)
    prev = -0.5 * np.log(1 - (ac[1] ** 2))
    for i in range(len(diffs)):
        curr = -0.5 * np.log(1 - (ac[i + 2] ** 2))
        diffs[i] = curr - prev
        prev = curr

    for i in range(len(diffs) - 1):
        if ((diffs[i] * diffs[i + 1]) < 0) and (diffs[i] < 0):
            return np.float64(i + 1)
    return np.float64(tau)


# ==================================================================
# Successive differences
# ==================================================================
@njit(fastmath=True, cache=True)
def md_hrv_classic_pnn40(y):
    """
    Proportion of successive differences exceeding 0.04σ (Mietus 2002)
    """
    a = len(y) - 1
    _sum = 0
    diffs = np.zeros(a, dtype=np.float64)
    for i in range(len(diffs)):
        diffs = abs(y[i + 1] - y[i]) * 1000
        if diffs > 40:
            _sum += 1
    _sum = _sum / a
    return _sum


@njit(fastmath=True, cache=True)
def sb_binarystats_diff_longstretch_0(y):
    """
    Longest period of successive incremental decreases
    """
    last_val = 0
    max_stretch = 0
    len_diff = len(y) - 1
    for i in range(len_diff):
        if ((y[i + 1] - y[i]) >= 0) or (i == (len_diff - 1)):
            stretch = i - last_val
            if stretch > max_stretch:
                max_stretch = stretch
            last_val = i

    return np.float64(max_stretch)


@njit(fastmath=True, cache=True)
def sb_motif_three_quantile_hh(y):
    """
    Shannon entropy of two successive letters in equiprobable 3-letter symbolization
    """
    indices = np.argsort(y)
    len_y = len(y)
    bins = np.zeros(len_y, dtype=np.int32)
    q1 = int(len_y / 3)
    q2 = int(q1 * 2)
    p = (np.zeros(q1, dtype=np.int32),
         np.zeros(q2 - q1, dtype=np.int32),
         np.zeros(len_y - q2, dtype=np.int32))

    # todo bug in TSML?
    for i in range(q1):
        bins[indices[i]] = 0
        p[0][i] = indices[i] + 1
    for i in range(q1, q2):
        bins[indices[i]] = 1
        p[1][i - q1] = indices[i] + 1
    for i in range(q2, len_y):
        bins[indices[i]] = 2
        p[2][i - q2] = indices[i] + 1

    _sum = 0.0
    for i in range(3):
        o = p[i]

        for n in range(3):
            sum2 = 0
            for k in range(len(o)):
                if (o[k] < len_y) and (bins[o[k]]) == n:
                    sum2 += 1

            if sum2 > 0:
                sum2 /= (len_y - 1)
                _sum += sum2 * np.log(sum2)

    return -_sum


@njit(fastmath=True, cache=True)
def fc_local_simple_mean1_tauresrat(y, ac):
    """
    Change in correlation length after iterative differencing
    """
    if (len(y) - 1) < 1:
        return 0.0

    res, mean_res = local_simple_mean(y, 1)
    len_res = len(res)

    length = np.float64(1 << int(np.round(np.log(len_res) / np.log(2))))
    length = int(length)
    if length < len_res:
        length *= 2

    fft = numba_fft_v(res - mean_res)
    res_ac = autocorr(res, fft)

    return np.float64(co_first_zero(res_ac) / co_first_zero(ac))


@njit(fastmath=True, cache=True)
def co_embed2_dist_tau_d_expfit_meandiff(y, ac):
    """
    Exponential fit to successive distances in 2-d embedding space
    """

    len_y = len(y)  # time series length
    tau = co_first_zero(ac)
    a = int(len_y / 10)
    if tau > a:
        tau = a

    len_d = len_y - tau - 1
    d = np.zeros(len_d, dtype=np.float64)
    d_mean = 0
    d_max = np.NINF
    d_min = np.PINF
    for i in range(len_d):
        n = np.sqrt((y[i + 1] - y[i]) ** 2 + (y[i + tau + 1] - y[i + tau]) ** 2)
        d[i] = n
        d_mean += n
        if n < d_min:
            d_min = n
        if n > d_max:
            d_max = n
    if d_mean == 0:
        return 0.0
    d_mean /= len_d
    d_range = d_max - d_min

    std = numba_std(d, d_mean)
    if (std == 0) or (d_range == 0):
        # if there is no change or there is only 1 point, std=0,
        # prevent division by zero
        num_bins = 1
        histogram = np.zeros(1, dtype=np.float64)
        bin_width = 0
    else:
        num_bins = np.ceil(d_range / (3.5 * std / len_d ** 0.3333333333333333))
        num_bins = int(num_bins)
        bin_width = d_range / num_bins

        if num_bins == 0:
            # was return nan
            return 0.0

        histogram = np.zeros(num_bins, dtype=np.float64)
        for val in d:
            idx = int((val - d_min) / bin_width)
            if idx >= num_bins:
                idx = num_bins - 1
            histogram[idx] += 1

    _sum = 0
    for i in range(num_bins):
        center = ((d_min + bin_width * i) * 2 + bin_width) / 2
        n = np.exp((-center) / d_mean) / d_mean
        if n < 0:
            n = 0
        _sum += np.abs(histogram[i] / len(d) - n)

    return _sum / num_bins


# ==================================================================
# Fluctuation Analysis
# ==================================================================
@njit(fastmath=True, cache=True)
def sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1(y):
    """
    Proportion of slower timescale fluctuations that scale with DFA (50% sampling)
    """
    len_y = len(y)
    len_cs = int(len_y / 2)
    cs = np.zeros(len_cs, dtype=np.float64)
    cs[0] = y[0]
    for i in range(1, len_cs):
        cs[i] = cs[i - 1] + y[i * 2]

    return fluct_prop(cs, len_y, True)


@njit(fastmath=True, cache=True)
def sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1(y):
    """
    Proportion of slower timescale fluctuations that scale with linearly rescaled range fits
    """
    len_y = len(y)
    cs = np.zeros_like(y)
    cs[0] = y[0]
    for i in range(1, len_y):
        cs[i] = cs[i - 1] + y[i]

    return fluct_prop(cs, len_y, False)


# ==================================================================
# Others
# ==================================================================
@njit(fastmath=True, cache=True)
def sb_transition_matrix_3ac_sum_diag_cov(y, ac):
    """
    Trace of covariance of transition matrix between symbols in 3-letter alphabet
    """
    len_y = len(y)
    tau = co_first_zero(ac)

    # downsample at rate 1:tau)
    ds_size = int((len_y - 1) / tau + 1)
    if ds_size == 1:
        return 0.0

    ds = np.zeros(ds_size, dtype=np.float64)
    for i in range(ds_size):
        ds[i] = y[i * tau]

    # (1) discretize
    indices = np.argsort(ds)
    bins = np.zeros(ds_size, dtype=np.int32)
    q1 = ds_size / 3
    q2 = q1 * 2
    q1 = int(np.ceil(q1))
    q2 = int(np.ceil(q2))
    # todo bug in tsml?
    for i in range(q1):
        bins[indices[i]] = 0
    for i in range(q1, q2):
        bins[indices[i]] = 1
    for i in range(q2, ds_size):
        bins[indices[i]] = 2

    # find 1-time transition matrix
    t = np.zeros((3, 3), dtype=np.float64)
    for i in range(ds_size - 1):
        t[bins[i + 1]][bins[i]] += 1

    means = np.zeros(3, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            t[i][j] /= (ds_size - 1)
            means[i] += t[i][j]
        means[i] /= 3

    # todo actually we could just compute the diagonal variances?
    sum = 0.0
    cov = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for n in range(i, 3):
            covariance = 0
            for j in range(3):
                covariance += (t[i, j] - means[i]) * (t[n, j] - means[n])
            covariance /= 2

            cov[i, n] = covariance
            cov[n, i] = covariance
        sum += cov[i][i]

    return sum


@njit(fastmath=True, cache=True)
def pd_periodicity_wang_th0_01(y):
    """
    Periodicity measure of (Wang et al. 2007)
    """
    len_y = len(y)
    if len_y < 4:
        # following matlab code
        return 0.0
    ySpline = splinefit(y)

    ySub = y - ySpline

    acmax = int(np.ceil(len_y / 3))
    acf = np.zeros(acmax, dtype=np.float64)

    for tau in range(1, acmax + 1):
        covariance = 0
        for i in range(len_y - tau):
            covariance += ySub[i] * ySub[i + tau]
        acf[tau - 1] = covariance / (len_y - tau)

    troughs = np.zeros(acmax, dtype=np.int32)
    peaks = np.zeros_like(troughs)
    nTroughs = 0
    nPeaks = 0
    for i in range(1, acmax - 1):
        slopeIn = acf[i] - acf[i - 1]
        slopeOut = acf[i + 1] - acf[i]
        if (slopeIn < 0) and (slopeOut > 0):
            troughs[nTroughs] = i
            nTroughs += 1
        elif (slopeIn > 0) and (slopeOut < 0):
            peaks[nPeaks] = i
            nPeaks += 1

    out = 0.0
    for i in range(nPeaks):
        iPeak = peaks[i]
        thePeak = acf[iPeak]

        j = -1
        while (troughs[j + 1] < iPeak) and (j + 1 < nTroughs):
            j += 1
        if j == -1:
            continue

        iTrough = troughs[j]
        theTrough = acf[iTrough]

        if (thePeak - theTrough) < 0.01:
            continue

        if thePeak < 0:
            continue

        out = iPeak
    return out
