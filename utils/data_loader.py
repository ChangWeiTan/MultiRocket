# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification
# https://arxiv.org/abs/2102.00457

import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

non_109_datasets = ["HandOutlines",
                    "NonInvasiveFetalECGThorax1",
                    "NonInvasiveFetalECGThorax2",
                    "AllGestureWiimoteX",
                    "AllGestureWiimoteY",
                    "AllGestureWiimoteZ",
                    "DodgerLoopDay",
                    "DodgerLoopGame",
                    "DodgerLoopWeekend",
                    "Fungi",
                    "GestureMidAirD1",
                    "GestureMidAirD2",
                    "GestureMidAirD3",
                    "GesturePebbleZ1",
                    "GesturePebbleZ2",
                    "MelbournePedestrian",
                    "PickupGestureWiimoteZ",
                    "PLAID",
                    "ShakeGestureWiimoteZ"]

classification_datasets = ["Adiac",  # 390,391,37,176,0.3887,0.3913 (3),0.3964
                           "ArrowHead",  # 36,175,3,251,0.2,0.2000 (0),0.2971
                           "Beef",  # 30,30,5,470,0.3333,0.3333 (0),0.3667
                           "BeetleFly",  # 20,20,2,512,0.25,0.3000 (7),0.3
                           "BirdChicken",  # 20,20,2,512,0.45,0.3000 (6),0.25
                           "Car",  # 60,60,4,577,0.2667,0.2333 (1),0.2667
                           "CBF",  # 30,900,3,128,0.1478,0.0044 (11),0.0033
                           "ChlorineConcentration",  # 467,3840,3,166,0.35,0.3500 (0),0.3516
                           "CinCECGTorso",  # 40,1380,4,1639,0.1029,0.0696 (1),0.3493
                           "Coffee",  # 28,28,2,286,0,0.0000 (0),0
                           "Computers",  # 250,250,2,720,0.424,0.3800 (12),0.3
                           "CricketX",  # 390,390,12,300,0.4231,0.2282 (10),0.2462
                           "CricketY",  # 390,390,12,300,0.4333,0.2410 (17),0.2564
                           "CricketZ",  # 390,390,12,300,0.4128,0.2538 (5),0.2462
                           "DiatomSizeReduction",  # 16,306,4,345,0.0654,0.0654 (0),0.0327
                           "DistalPhalanxOutlineAgeGroup",  # 400,139,3,80,0.3741,0.3741 (0),0.2302
                           "DistalPhalanxOutlineCorrect",  # 600,276,2,80,0.2826,0.2754 (1),0.2826
                           "DistalPhalanxTW",  # 400,139,6,80,0.3669,0.3669 (0),0.4101
                           "Earthquakes",  # 322,139,2,512,0.2878,0.2734 (6),0.2806
                           "ECG200",  # 100,100,2,96,0.12,0.1200 (0),0.23
                           "ECG5000",  # 500,4500,5,140,0.0751,0.0749 (1),0.0756
                           "ECGFiveDays",  # 23,861,2,136,0.2033,0.2033 (0),0.2323
                           "ElectricDevices",  # 8926,7711,7,96,0.4492,0.3806 (14),0.3988
                           "FaceAll",  # 560,1690,14,131,0.2864,0.1917 (3),0.1923
                           "FaceFour",  # 24,88,4,350,0.2159,0.1136 (2),0.1705
                           "FacesUCR",  # 200,2050,14,131,0.2307,0.0878 (12),0.0951
                           "FiftyWords",  # 450,455,50,270,0.3692,0.2418 (6),0.3099
                           "Fish",  # 175,175,7,463,0.2171,0.1543 (4),0.1771
                           "FordA",  # 3601,1320,2,500,0.3348,0.3091 (1),0.4455
                           "FordB",  # 3636,810,2,500,0.3938,0.3926 (1),0.3802
                           "GunPoint",  # 50,150,2,150,0.0867,0.0867 (0) ,0.0933
                           "Ham",  # 109,105,2,431,0.4,0.4000 (0),0.5333
                           "HandOutlines",  # 1000,370,2,2709,0.1378,0.1378 (0),0.1189
                           "Haptics",  # 155,308,5,1092,0.6299,0.5877 (2),0.6234
                           "Herring",  # 64,64,2,512,0.4844,0.4688 (5),0.4688
                           "InlineSkate",  # 100,550,7,1882,0.6582,0.6127 (14),0.6164
                           "InsectWingbeatSound",  # 220,1980,11,256,0.4384,0.4152 (1),0.6449
                           "ItalyPowerDemand",  # 67,1029,2,24,0.0447,0.0447 (0),0.0496
                           "LargeKitchenAppliances",  # 375,375,3,720,0.5067,0.2053 (94),0.2053
                           "Lightning2",  # 60,61,2,637,0.2459,0.1311 (6),0.1311
                           "Lightning7",  # 70,73,7,319,0.4247,0.2877 (5),0.274
                           "Mallat",  # 55,2345,8,1024,0.0857,0.0857 (0),0.0661
                           "Meat",  # 60,60,3,448,0.0667,0.0667 (0),0.0667
                           "MedicalImages",  # 381,760,10,99,0.3158,0.2526 (20),0.2632
                           "MiddlePhalanxOutlineAgeGroup",  # 400,154,3,80,0.4805,0.4805 (0),0.5
                           "MiddlePhalanxOutlineCorrect",  # 600,291,2,80,0.2337,0.2337 (0),0.3024
                           "MiddlePhalanxTW",  # 399,154,6,80,0.487,0.4935 (3),0.4935
                           "MoteStrain",  # 20,1252,2,84,0.1214,0.1342 (1),0.1653
                           "NonInvasiveFetalECGThorax1",  # 1800,1965,42,750,0.171,0.1893 (1),0.2097
                           "NonInvasiveFetalECGThorax2",  # 1800,1965,42,750,0.1201,0.1290 (1),0.1354
                           "OliveOil",  # 30,30,4,570,0.1333,0.1333 (0),0.1667
                           "OSULeaf",  # 200,242,6,427,0.4793,0.3884 (7),0.4091
                           "PhalangesOutlinesCorrect",  # 1800,858,2,80,0.2389,0.2389 (0),0.2716
                           "Phoneme",  # 214,1896,39,1024,0.8908,0.7727 (14),0.7716
                           "Plane",  # 105,105,7,144,0.0381,0.0000 (5),0
                           "ProximalPhalanxOutlineAgeGroup",  # 400,205,3,80,0.2146,0.2146 (0),0.1951
                           "ProximalPhalanxOutlineCorrect",  # 600,291,2,80,0.1924,0.2096 (1),0.2165
                           "ProximalPhalanxTW",  # 400,205,6,80,0.2927,0.2439 (2),0.2439
                           "RefrigerationDevices",  # 375,375,3,720,0.6053,0.5600 (8),0.536
                           "ScreenType",  # 375,375,3,720,0.64,0.5893 (17),0.6027
                           "ShapeletSim",  # 20,180,2,500,0.4611,0.3000 (3),0.35
                           "ShapesAll",  # 600,600,60,512,0.2483,0.1980 (4),0.2317
                           "SmallKitchenAppliances",  # 375,375,3,720,0.6587,0.3280 (15),0.3573
                           "SonyAIBORobotSurface1",  # 20,601,2,70,0.3045,0.3045 (0),0.2745
                           "SonyAIBORobotSurface2",  # 27,953,2,65,0.1406,0.1406 (0),0.1689
                           "StarLightCurves",  # 1000,8236,3,1024,0.1512,0.0947 (16),0.0934
                           "Strawberry",  # 613,370,2,235,0.0541,0.0541 (0),0.0595
                           "SwedishLeaf",  # 500,625,15,128,0.2112,0.1536 (2),0.208
                           "Symbols",  # 25,995,6,398,0.1005,0.0623 (8),0.0503
                           "SyntheticControl",  # 300,300,6,60,0.12,0.0167 (6),0.0067
                           "ToeSegmentation1",  # 40,228,2,277,0.3202,0.2500 (8),0.2281
                           "ToeSegmentation2",  # 36,130,2,343,0.1923,0.0923 (5),0.1615
                           "Trace",  # 100,100,4,275,0.24,0.0100 (3),0
                           "TwoLeadECG",  # 23,1139,2,82,0.2529,0.1317 (4),0.0957
                           "TwoPatterns",  # 1000,4000,4,128,0.0932,0.0015 (4),0
                           "UWaveGestureLibraryAll",  # 896,3582,8,945,0.0519,0.0343 (4),0.1083
                           "UWaveGestureLibraryX",  # 896,3582,8,315,0.2607,0.2267 (4),0.2725
                           "UWaveGestureLibraryY",  # 896,3582,8,315,0.338,0.3009 (4),0.366
                           "UWaveGestureLibraryZ",  # 896,3582,8,315,0.3504,0.3222 (6),0.3417
                           "Wafer",  # 1000,6164,2,152,0.0045,0.0045 (1),0.0201
                           "Wine",  # 57,54,2,234,0.3889,0.3889 (0),0.4259
                           "WordSynonyms",  # 267,638,25,270,0.3824,0.2618 (9),0.3511
                           "Worms",  # 181,77,5,900,0.5455,0.4675 (9),0.4156
                           "WormsTwoClass",  # 181,77,2,900,0.3896,0.4156 (7),0.3766
                           "Yoga",  # 300,3000,2,426,0.1697,0.1560 (7),0.1637
                           "ACSF1",  # 100,100,10,1460,0.46,0.3800 (4),0.36
                           "AllGestureWiimoteX",  # 300,700,10,Vary,0.4843, 0.2829 (14),0.2843
                           "AllGestureWiimoteY",  # 300,700,10,Vary,0.4314, 0.2700 (9),0.2714
                           "AllGestureWiimoteZ",  # 300,700,10,Vary,0.5457,0.3486 (11),0.3571
                           "BME",  # 30,150,3,128,0.1667,0.0200 (4),0.1
                           "Chinatown",  # 20,345,2,24,0.0464,0.0464 (0),0.0435
                           "Crop",  # 7200,16800,24,46,0.2883,0.2883 (0),0.3348
                           "DodgerLoopDay",  # 78,80,7,288,0.45, 0.4125 (1),0.5
                           "DodgerLoopGame",  # 20,138,2,288,0.1159, 0.0725 (1),0.1232
                           "DodgerLoopWeekend",  # 20,138,2,288,0.0145, 0.0217 (1),0.0507
                           "EOGHorizontalSignal",  # 362,362,12,1250,0.5829, 0.5249 (1),0.4972
                           "EOGVerticalSignal",  # 362,362,12,1250,0.558, 0.5249 (2),0.5525
                           "EthanolLevel",  # 504,500,4,1751,0.726,0.7180 (1),0.724
                           "FreezerRegularTrain",  # 150,2850,2,301,0.1951,0.0930 (1),0.1011
                           "FreezerSmallTrain",  # 28,2850,2,301,0.3302,0.3302 (0),0.2467
                           "Fungi",  # 18,186,18,201,0.1774,0.1774 (0),0.1613
                           "GestureMidAirD1",  # 208,130,26,Vary,0.4231, 0.3615 (5),0.4308
                           "GestureMidAirD2",  # 208,130,26,Vary,0.5077, 0.4000 (6),0.3923
                           "GestureMidAirD3",  # 208,130,26,Vary,0.6538, 0.6231 (1),0.6769
                           "GesturePebbleZ1",  # 132,172,6,Vary,0.2674,0.1744 (2),0.2093
                           "GesturePebbleZ2",  # 146,158,6,Vary,0.3291,0.2215 (6),0.3291
                           "GunPointAgeSpan",  # 135,316,2,150,0.1013,0.0348 (3),0.0823
                           "GunPointMaleVersusFemale",  # 135,316,2,150,0.0253,0.0253 (0),0.0032
                           "GunPointOldVersusYoung",  # 135,316,2,150,0.0476,0.0349 (4),0.1619
                           "HouseTwenty",  # 40,119,2,2000,0.3361, 0.0588 (33),0.0756
                           "InsectEPGRegularTrain",  # 62,249,3,601,0.3213,0.1727 (11),0.1285
                           "InsectEPGSmallTrain",  # 17,249,3,601,0.3373,0.3052 (1),0.2651
                           "MelbournePedestrian",  # 1200,2450,10,24,0.1518,0.1518 (0),0.2094
                           "MixedShapesRegularTrain",  # 500,2425,5,1024,0.1027, 0.0911 (4),0.1584
                           "MixedShapesSmallTrain",  # 100,2425,5,1024,0.1645, 0.1674 (7),0.2202
                           "PickupGestureWiimoteZ",  # 50,50,10,Vary,0.44,0.3400 (17),0.34
                           "PigAirwayPressure",  # 104,208,52,2000,0.9423,0.9038 (1),0.8942
                           "PigArtPressure",  # 104,208,52,2000,0.875,0.8029 (1),0.7548
                           "PigCVP",  # 104,208,52,2000,0.9183,0.8413 (11),0.8462
                           "PLAID",  # 537,537,11,Vary,0.4786,0.1862 (3),0.1601
                           "PowerCons",  # 180,180,2,144,0.0667,0.0778 (3),0.1222
                           "Rock",  # 20,50,4,2844,0.16, 0.1600 (0),0.4
                           "SemgHandGenderCh2",  # 300,600,2,1500,0.2383,0.1550 (1),0.1983
                           "SemgHandMovementCh2",  # 450,450,6,1500,0.6311,0.3622 (1),0.4156
                           "SemgHandSubjectCh2",  # 450,450,5,1500,0.5956,0.2000 (3),0.2733
                           "ShakeGestureWiimoteZ",  # 50,50,10,Vary,0.4,0.1600 (6),0.14
                           "SmoothSubspace",  # 150,150,3,15,0.0933,0.0533 (1),0.1733
                           "UMD",  # 36,144,3,150,0.2361,0.0278 (6),0.0069
                           ]


def get_classification_datasets_summary(dataset=None, subset="full"):
    if subset == "109":
        if os.path.exists("../data/classification_datasets_109.csv"):
            df = pd.read_csv("../data/classification_datasets_109.csv")
        else:
            df = pd.read_csv(os.getcwd() + "/data/classification_datasets_109.csv")
        df.columns = [x.strip() for x in df.columns]
        if dataset is None:
            return df
    elif subset == "bakeoff":
        if os.path.exists("../data/classification_datasets_bakeoff.csv"):
            df = pd.read_csv("../data/classification_datasets_bakeoff.csv")
        else:
            df = pd.read_csv(os.getcwd() + "/data/classification_datasets_bakeoff.csv")
        df.columns = [x.strip() for x in df.columns]
        if dataset is None:
            return df
    elif subset == "development":
        if os.path.exists("../data/classification_datasets_development.csv"):
            df = pd.read_csv("../data/classification_datasets_development.csv")
        else:
            df = pd.read_csv(os.getcwd() + "/data/classification_datasets_development.csv")
        df.columns = [x.strip() for x in df.columns]
        if dataset is None:
            return df
    elif subset == "holdout":
        if os.path.exists("../data/classification_datasets_development.csv"):
            df_dev = pd.read_csv("../data/classification_datasets_development.csv")
        else:
            df_dev = pd.read_csv(os.getcwd() + "/data/classification_datasets_development.csv")
        if os.path.exists("../data/classification_datasets_bakeoff.csv"):
            df = pd.read_csv("../data/classification_datasets_bakeoff.csv")
        else:
            df = pd.read_csv(os.getcwd() + "/data/classification_datasets_bakeoff.csv")
        df = df.loc[~df["Name"].isin(df_dev["Name"])].reset_index(drop=True)
        df.columns = [x.strip() for x in df.columns]
        if dataset is None:
            return df
    else:
        if os.path.exists("../data/classification_datasets.csv"):
            df = pd.read_csv("../data/classification_datasets.csv")
        else:
            df = pd.read_csv(os.getcwd() + "/data/classification_datasets.csv")
        df.columns = [x.strip() for x in df.columns]
        if dataset is None:
            return df

    return df.loc[df.Name == dataset].reset_index(drop=True)


def read_univariate_ucr(filename, normalise=True):
    data = np.loadtxt(filename, delimiter='\t')
    Y = data[:, 0]
    X = data[:, 1:]

    scaler = StandardScaler()
    for i in range(len(X)):
        for j in range(len(X[i])):
            if np.isnan(X[i, j]):
                X[i, j] = random.random() / 1000
        # scale it later
        if normalise:
            tmp = scaler.fit_transform(X[i].reshape(-1, 1))
            X[i] = tmp[:, 0]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, Y


def fill_missing(x: np.array,
                 max_len: int,
                 vary_len: str = "suffix-noise",
                 normalise: bool = True):
    if vary_len == "zero":
        if normalise:
            x = StandardScaler().fit_transform(x)
        x = np.nan_to_num(x)
    elif vary_len == 'prefix-suffix-noise':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)
            diff_len = int(0.5 * (max_len - seq_len))

            for j in range(diff_len):
                x[i, j] = random.random() / 1000

            for j in range(diff_len, seq_len):
                x[i, j] = series[j - seq_len]

            for j in range(seq_len, max_len):
                x[i, j] = random.random() / 1000

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    elif vary_len == 'uniform-scaling':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)

            for j in range(max_len):
                scaling_factor = int(j * seq_len / max_len)
                x[i, j] = series[scaling_factor]
            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    else:
        for i in range(len(x)):
            for j in range(len(x[i])):
                if np.isnan(x[i, j]):
                    x[i, j] = random.random() / 1000

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]

    return x


def process_ts_data(X,
                    vary_len: str = "suffix-noise",
                    normalise: bool = False):
    """
    This is a function to process the data, i.e. convert dataframe to numpy array
    :param X:
    :param normalise:
    :return:
    """
    num_instances, num_dim = X.shape
    columns = X.columns
    max_len = np.max([len(X[columns[0]][i]) for i in range(num_instances)])
    output = np.zeros((num_instances, num_dim, max_len), dtype=np.float64)

    for i in range(num_dim):
        for j in range(num_instances):
            output[j, i, :] = X[columns[i]][j].values
        output[:, i, :] = fill_missing(
            output[:, i, :],
            max_len,
            vary_len,
            normalise
        )

    return output
