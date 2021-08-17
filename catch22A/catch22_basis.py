# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Effective summary statistics for convolutional outputs in time series classification
# https://arxiv.org/abs/2102.00457

import numpy as np
from numba import njit

# =======================================================================================================
# Catch22 basis Operations
# =======================================================================================================
from utils.tools import numba_linear_regression


@njit(fastmath=True, cache=True)
def co_first_zero(ac):
    for i in range(1, len(ac)):
        if ac[i] < 0:
            return i

    return len(ac)


@njit(fastmath=True, cache=True)
def outlier_include(y):
    total = 0
    threshold = 0
    for v in y:
        if v >= 0:
            total += 1
            if v > threshold:
                threshold = v

    if threshold < 0.01:
        return 0.0

    num_thresholds = int(threshold / 0.01) + 1
    means = np.zeros(num_thresholds, dtype=np.float64)
    dists = np.zeros_like(means)
    medians = np.zeros_like(means)
    for i in range(num_thresholds):
        d = i * 0.01

        flag = np.zeros(len(y), dtype=np.int32)
        r = np.zeros(len(y), dtype=np.float64)
        len_r = 0
        for n in range(len(y)):
            if y[n] >= d:
                r[n] = n + 1.0
                flag[n] = 1
                len_r += 1
        r = r[np.nonzero(flag)]

        if len_r == 0:
            continue

        diff = np.zeros(len_r - 1, dtype=np.float64)
        for n in range(len(diff)):
            diff[n] = r[n + 1] - r[n]
            means[i] += diff[n]

        if len(diff) > 0:
            means[i] /= len(diff)
        dists[i] = len(diff) * 100 / total
        medians[i] = np.median(r) / (len(y) / 2) - 1

    mj = 0
    fbi = num_thresholds - 1
    for i in range(num_thresholds):
        if dists[i] > 2:
            mj = i
        if np.isnan(means[i]):
            fbi = num_thresholds - 1 - i

    trim_limit = mj
    if mj < fbi:
        trim_limit = fbi

    return np.median(medians[:(trim_limit + 1)])


@njit(fastmath=True, cache=True)
def histogram_mode(y, y_min, y_max, num_bins):
    """
    DN_HistogramMode      Mode of a data vector.
    Measures the mode of the data vector using histograms with a given number of bins
    """
    bin_width = (y_max - y_min) / num_bins
    histogram = np.zeros(num_bins, dtype=np.float64)
    for i in y:
        if bin_width > 0:
            idx = int((i - y_min) / bin_width)
        else:
            idx = 0
        if idx >= num_bins:
            idx = num_bins - 1
        histogram[idx] += 1

    bin_edges = np.zeros(num_bins + 1, dtype=np.float64)
    for i in range(len(bin_edges)):
        bin_edges[i] = i * bin_width + y_min

    # Compute bin centers from bin edges:
    # binCenters = mean([binEdges(1:end-1); binEdges(2:end)]);
    # Mean position of maximums( if multiple):
    max_count = 0
    num_maxs = 1
    max_sum = 0
    for i in range(num_bins):
        if histogram[i] > max_count:
            max_count = histogram[i]
            num_maxs = 1
            max_sum = (bin_edges[i] + bin_edges[i + 1]) * 0.5
        elif histogram[i] == max_count:
            num_maxs += 1
            max_sum += (bin_edges[i] + bin_edges[i + 1]) * 0.5

    return max_sum / num_maxs


@njit(fastmath=True, cache=True)
def splinefit(y):
    """
    %SPLINEFIT Fit a spline to noisy data.
    %   PP = SPLINEFIT(X,Y,BREAKS) fits a piecewise cubic spline with breaks
    %   (knots) BREAKS to the noisy data (X,Y). X is a vector and Y is a vector
    %   or an ND array. If Y is an ND array, then X(j) and Y(:,...,:,j) are
    %   matched. Use PPVAL to evaluate PP.
    %
    %   PP = SPLINEFIT(X,Y,P) where P is a positive integer interpolates the
    %   breaks linearly from the sorted locations of X. P is the number of
    %   spline pieces and P+1 is the number of breaks.
    %
    %   OPTIONAL INPUT
    %   Argument places 4 to 8 are reserved for optional input.
    %   These optional arguments can be given in any order:
    %
    %   PP = SPLINEFIT(...,'p') applies periodic boundary conditions to
    %   the spline. The period length is MAX(BREAKS)-MIN(BREAKS).
    %
    %   PP = SPLINEFIT(...,'r') uses robust fitting to reduce the influence
    %   from outlying data points. Three iterations of weighted least squares
    %   are performed. Weights are computed from previous residuals.
    %
    %   PP = SPLINEFIT(...,BETA), where 0 < BETA < 1, sets the robust fitting
    %   parameter BETA and activates robust fitting ('r' can be omitted).
    %   Default is BETA = 1/2. BETA close to 0 gives all data equal weighting.
    %   Increase BETA to reduce the influence from outlying data. BETA close
    %   to 1 may cause instability or rank deficiency.
    %
    %   PP = SPLINEFIT(...,N) sets the spline order to N. Default is a cubic
    %   spline with order N = 4. A spline with P pieces has P+N-1 degrees of
    %   freedom. With periodic boundary conditions the degrees of freedom are
    %   reduced to P.
    %
    %   PP = SPLINEFIT(...,CON) applies linear constraints to the spline.
    %   CON is a structure with fields 'xc', 'yc' and 'cc':
    %       'xc', x-locations (vector)
    %       'yc', y-values (vector or ND array)
    %       'cc', coefficients (matrix).
    %
    %   Constraints are linear combinations of derivatives of order 0 to N-2
    %   according to
    %
    %     cc(1,j)*y(x) + cc(2,j)*y'(x) + ... = yc(:,...,:,j),  x = xc(j).
    %
    %   The maximum number of rows for 'cc' is N-1. If omitted or empty 'cc'
    %   defaults to a single row of ones. Default for 'yc' is a zero array.
    %
    %   EXAMPLES
    %
    %       % Noisy data
    %       x = linspace(0,2*pi,100);
    %       y = sin(x) + 0.1*randn(size(x));
    %       % Breaks
    %       breaks = [0:5,2*pi];
    %
    %       % Fit a spline of order 5
    %       pp = splinefit(x,y,breaks,5);
    %
    %       % Fit a spline of order 3 with periodic boundary conditions
    %       pp = splinefit(x,y,breaks,3,'p');
    %
    %       % Constraints: y(0) = 0, y'(0) = 1 and y(3) + y"(3) = 0
    %       xc = [0 0 3];
    %       yc = [0 1 0];
    %       cc = [1 0 1; 0 1 0; 0 0 1];
    %       con = struct('xc',xc,'yc',yc,'cc',cc);
    %
    %       % Fit a cubic spline with 8 pieces and constraints
    %       pp = splinefit(x,y,8,con);
    %
    %       % Fit a spline of order 6 with constraints and periodicity
    %       pp = splinefit(x,y,breaks,con,6,'p');
    %
    %   See also SPLINE, PPVAL, PPDIFF, PPINT

    %   Author: Jonas Lundgren <splinefit@gmail.com> 2010

    %   2009-05-06  Original SPLINEFIT.
    %   2010-06-23  New version of SPLINEFIT based on B-splines.
    %   2010-09-01  Robust fitting scheme added.
    %   2010-09-01  Support for data containing NaNs.
    %   2011-07-01  Robust fitting parameter added.
    :param y:
    :return:
    """
    len_y = len(y)
    # n = 4
    # deg = 3  # n always = 4
    # beta = 0
    # dim = 1
    # pieces0 = 2  # number of pieces. always 2 since breaks0 is always initialised to 3

    breaks = np.array((0, len_y / 2 - 1, len_y - 1), dtype=np.float64)
    h0 = np.array((breaks[1] - breaks[0], breaks[2] - breaks[1]), dtype=np.float64)  # spacing

    # distances between knots to replicate points to the sides
    # hcopy = repmat(h0,ceil(deg/pieces0),1); # concatenate h0 by ceil(deg/pieces0)
    # np.ceil(deg / pieces0, 1) always = 2 and h0 always has 2 elements
    hcopy = np.array((h0[0], h0[1], h0[0], h0[1]), dtype=np.float64)

    # to the left
    # hl = hcopy(end:-1:end-deg+1);
    # bl = breaks0(1) - cumsum(hl);
    a = hcopy[3]
    b = hcopy[2] + a
    c = hcopy[1] + b
    bl = np.array((breaks[0] - a,
                   breaks[0] - b,
                   breaks[0] - c),
                  dtype=np.int32)

    # and to the right
    # hr = hcopy(1:deg);
    # br = breaks0(end) + cumsum(hr);
    a = hcopy[0]
    b = hcopy[1] + a
    c = hcopy[2] + b
    br = np.array((breaks[2] + a,
                   breaks[2] + b,
                   breaks[2] + c),
                  dtype=np.int32)

    # add breaks
    breaksExt = np.zeros(9, dtype=np.int32)
    for i in range(3):
        breaksExt[i] = bl[2 - i]
        breaksExt[i + 3] = breaks[i]
        breaksExt[i + 6] = br[i]

    hExt = np.zeros(8, dtype=np.int32)
    for i in range(8):
        hExt[i] = breaksExt[i + 1] - breaksExt[i]

    # Initiate polynomial coefficients
    # n*pieces always = 32, n = 4, pieces = 8
    coefs = np.zeros((32, 4), dtype=np.float64)
    for i in range(0, 32, 4):
        coefs[i, 0] = 1

    # Expand h
    # ii = [1:pieces; ones(deg,pieces)];
    # ii = cumsum(ii,1);
    # ii = min(ii,pieces);
    ii = np.zeros((4, 8), dtype=np.int32)
    for i in range(8):
        ii[0][i] = i if i < 7 else 7
        ii[1][i] = (i + 1) if (i + 1) < 7 else 7
        ii[2][i] = (i + 2) if (i + 2) < 7 else 7
        ii[3][i] = (i + 3) if (i + 3) < 7 else 7

        # ii[0][i] = numba_min(i, 7)
        # ii[1][i] = numba_min(i + 1, 7)
        # ii[2][i] = numba_min(i + 2, 7)
        # ii[3][i] = numba_min(i + 3, 7)

    # H = h(ii(:));
    H = np.zeros(32, dtype=np.float64)
    for i in range(32):
        a = int(i % 4)
        b = int(i / 4)
        H[i] = hExt[ii[a, b]]

    Q = np.zeros((4, 8), dtype=np.float64)
    fmax = np.zeros(32, dtype=np.float64)
    for k in range(1, 4):
        for j in range(k):
            for i in range(32):
                coefs[i, j] = coefs[i, j] * H[i] / (k - j)

        for i in range(32):
            for j in range(4):
                a = int(i % 4)
                b = int(i / 4)
                Q[a, b] += coefs[i, j]

        for i in range(8):
            for j in range(1, 4):
                Q[j, i] += Q[j - 1][i]

        for i in range(32):
            a = int(i % 4)
            b = int(i / 4)
            if a > 0:
                coefs[i, k] = Q[a - 1][b]

        for i in range(8):
            for j in range(4):
                a = int(i * 4 + j)
                fmax[a] = Q[3][i]

        for i in range(k + 1):
            for j in range(32):
                coefs[j, i] /= fmax[j]

        for i in range(29):
            for j in range(k + 1):
                coefs[i, j] -= coefs[3 + i, j]

        for i in range(0, 32, 4):
            coefs[i, k] = 0

    for k in range(3):
        for i in range(32):
            coefs[i, 3 - (k + 1)] /= H[i]

    jj = np.zeros((4, 2), dtype=np.int32)
    for i in range(4):
        for j in range(2):
            if i == 0:
                jj[i, j] = 4 * (1 + j)
            else:
                jj[i, j] = 3

    for i in range(1, 4):
        for j in range(2):
            jj[i, j] += jj[i - 1, j]

    coefsOut = np.zeros((8, 4), dtype=np.float64)
    for i in range(8):
        a = int(i % 4)
        b = int(i / 4)
        jj_flat = jj[a, b] - 1
        for j in range(4):
            coefsOut[i, j] = coefs[jj_flat, j]

    xsB = np.zeros(len_y * 4, dtype=np.int32)
    indexB = np.zeros_like(xsB)
    breakInd = 1
    for i in range(len_y):
        if (i >= breaks[1]) and (breakInd < 2):
            breakInd += 1
        for j in range(4):
            a = int(i * 4 + j)
            b = breakInd - 1
            xsB[a] = i - breaks[b]
            indexB[a] = j + b * 4

    vB = np.zeros(len(xsB), dtype=np.float64)
    for i in range(len(xsB)):
        vB[i] = coefsOut[indexB[i], 0]

    for i in range(1, 4):
        for j in range(len(xsB)):
            vB[j] = vB[j] * xsB[j] + coefsOut[indexB[j], i]

    A = np.zeros(len_y * 5, dtype=np.float64)
    breakInd = 0
    for i in range(len(xsB)):
        tmp = int(i / 4)
        if tmp >= breaks[1]:
            breakInd = 1
        idx = int(int(i % 4) + breakInd + tmp * 5)
        A[idx] = vB[i]

    x = np.zeros(5, dtype=np.float64)
    # lsqsolve_sub(size, n+1, A, size, y, x);
    # (const int sizeA1, const int sizeA2, const double *A, const int sizeb, const
    # double *b, double *x)
    AT = np.zeros(len(A), dtype=np.float64)
    ATA = np.zeros(25, dtype=np.float64)
    ATb = np.zeros(5, dtype=np.float64)
    for i in range(len_y):
        for j in range(5):
            idx1 = int(j * len_y + i)
            idx2 = int(i * 5 + j)
            AT[idx1] = A[idx2]

    # matrix_multiply(sizeA2, sizeA1, AT, sizeA1, sizeA2, A, ATA);
    # (const int sizeA1, const int sizeA2, const double *A, const int sizeB1, const
    # int sizeB2, const double *B, double *C)
    for i in range(5):
        for j in range(5):
            for k in range(len_y):
                idx1 = int(i * 5 + j)
                idx2 = int(i * len_y + k)
                idx3 = int(k * 5 + j)
                ATA[idx1] += AT[idx2] * A[idx3]

    # matrix_times_vector(sizeA2, sizeA1, AT, sizeA1, b, ATb);
    # (const int sizeA1, const int sizeA2, const double *A, const int sizeb, const
    # double *b, double *c)
    for i in range(5):
        for k in range(len_y):
            idx = int(i * len_y + k)
            ATb[i] += AT[idx] * y[k]

    # gauss_elimination(sizeA2, ATA, ATb, x);
    # (int size, double *A, double *b, double *x)
    AElim = np.zeros((5, 5), dtype=np.float64)
    bElim = np.zeros(5, dtype=np.float64)

    for i in range(5):
        for j in range(5):
            idx = int(i * 5 + j)
            AElim[i, j] = ATA[idx]
        bElim[i] = ATb[i]

    for i in range(5):
        for j in range(i + 1, 5):
            factor = AElim[j, i] / AElim[i, i]
            bElim[j] -= factor * bElim[i]

            for k in range(i, 5):
                AElim[j, k] -= factor * AElim[i, k]

    for i in range(4, -1, -1):
        bMinusATemp = bElim[i]
        for j in range(i + 1, 5):
            bMinusATemp -= x[j] * AElim[i, j]
        x[i] = bMinusATemp / AElim[i, i]

    C = np.zeros((5, 8), dtype=np.float64)
    for i in range(32):
        CRow = int(i % 4 + (i / 4) % 2)
        CCol = int(i / 4)

        coefRow = int(i % 8)
        coefCol = int(i / 8)

        C[CRow, CCol] = coefsOut[coefRow, coefCol]

    coefsSpline = np.zeros((2, 4), dtype=np.float64)
    for j in range(8):
        coefCol = int(j / 2)
        coefRow = int(j % 2)
        for i in range(5):
            coefsSpline[coefRow, coefCol] += C[i, j] * x[i]

    yOut = np.zeros(len_y, dtype=np.float64)

    for i in range(1, 4):
        for j in range(len_y):
            secondHalf = 1
            if j < breaks[1]:
                secondHalf = 0
            # yOut[j] = yOut[j] * (j - breaks[1] * secondHalf) + coefsSpline[secondHalf, i]
            yOut[j] = coefsSpline[secondHalf, 0] * (j - breaks[1] * secondHalf) + coefsSpline[secondHalf, i]

    return yOut


@njit(fastmath=True, cache=True)
def summaries_welth_rect(y, centroid, fft):
    len_y = len(y)
    new_len = int(len(fft) / 2 + 1)
    p = np.zeros(new_len, dtype=np.float64)
    cs = np.zeros_like(p)

    p[0] = ((np.abs(fft[0]) ** 2) / len_y) / (2 * np.pi)
    cs[0] = p[0]
    for i in range(1, new_len - 1):
        p[i] = (((np.abs(fft[i]) ** 2) / len_y) * 2) / (2 * np.pi)
        cs[i] = cs[i - 1] + p[i]

    p[-1] = ((np.abs(fft[-1]) ** 2) / len_y) / (2 * np.pi)
    cs[-1] = cs[-2] + p[-1]

    w = np.zeros_like(p)
    for i in range(new_len):
        w[i] = i * (1 / len(fft)) * np.pi * 2

    if centroid:
        threshold = cs[-1] / 2
        for i in range(new_len):
            if cs[i] > threshold:
                return w[i]
        return 0.0
    else:
        tau = int(np.floor(new_len / 5))
        sum = cs[tau - 1]
        return sum * (w[1] - w[0])


@njit(fastmath=True, cache=True)
def local_simple_mean(y, train_len):
    len_res = len(y) - train_len
    res = np.zeros(len_res, dtype=np.float64)
    mean_res = 0
    for i in range(len_res):
        sum = 0
        for n in range(train_len):
            sum += y[i + n]
        res[i] = y[i + train_len] - sum / train_len
        mean_res += res[i]
    mean_res /= len_res
    return res, mean_res


@njit(fastmath=True, cache=True)
def fluct_prop(y, og_length, dfa):
    len_y = len(y)
    a = [5]
    _min = np.log(5)
    _max = np.log(og_length / 2)
    inc = (_max - _min) / 49
    nTau = 1
    for i in range(1, 50):
        val = int(np.round(np.exp(_min + inc * i)))
        if val != a[-1]:
            a.append(val)
            nTau += 1

    if nTau < 12:
        # print("Return NaN from fluct_prop with " + str(nTau))
        return 0.0

    f = np.zeros(nTau, dtype=np.float64)
    for i in range(nTau):
        tau = int(a[i])
        buffSize = int(len_y / tau)
        lag = 0
        if buffSize == 0:
            buffSize = 1
            lag = 1

        buffer = np.zeros((buffSize, tau), dtype=np.float64)
        count = 0
        for n in range(buffSize):
            for j in range(tau - lag):
                buffer[n, j] = y[count]
                count += 1

        d = np.zeros(tau, dtype=np.float64)
        for n in range(tau):
            d[n] = n + 1

        for n in range(buffSize):
            co = numba_linear_regression(d, buffer[n], tau, 0)

            buffer_max = np.NINF
            buffer_min = np.PINF
            for j in range(tau):
                buffer[n, j] = buffer[n, j] - (co[0] * (j + 1) + co[1])
                if buffer[n, j] < buffer_min:
                    buffer_min = buffer[n, j]
                if buffer[n, j] > buffer_max:
                    buffer_max = buffer[n, j]
            if dfa:
                for j in range(tau):
                    f[i] += buffer[n, j] * buffer[n, j]
            else:
                f[i] += (buffer_max - buffer_min) ** 2

        if dfa:
            f[i] = np.sqrt(f[i] / (buffSize * tau))
        else:
            f[i] = np.sqrt(f[i] / buffSize)

    logA = np.zeros(nTau, dtype=np.float64)
    logF = np.zeros(nTau, dtype=np.float64)
    for i in range(nTau):
        logA[i] = np.log(a[i])
        logF[i] = np.log(f[i])

    nsser = int(nTau - 11)
    index_of_min = 0
    sser_min = np.PINF
    sserr = np.zeros(nsser, dtype=np.float64)
    for i in range(6, nTau - 5):
        co = numba_linear_regression(logA, logF, i, 0)
        co2 = numba_linear_regression(logA, logF, nTau - i + 1, i - 1)

        sum1 = 0
        for n in range(i):
            sum1 += (logA[n] * co[0] + co[1] - logF[n]) ** 2

        sum2 = 0
        for n in range(nTau - i + 1):
            sum2 += (logA[n + i - 1] * co2[0] + co2[1] - logF[n + i - 1]) ** 2

        sserr[i - 6] = np.sqrt(sum1) + np.sqrt(sum2)
        if sserr[i - 6] < sser_min:
            sser_min = sserr[i - 6]
            index_of_min = i - 6

    return (index_of_min + 6) / nTau
