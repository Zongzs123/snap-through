import numpy as np
import pandas as pd
from keras import backend as K


# define a simple dataloader
def load_data(path):
    columns = ["L", "w", "w^3", "alpha", "num_vert", "num_hori", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9",
               "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24",
               "s25", "s26", "s27", "s28", "s29", "s30", "s31"]
    filename = path
    data1_1 = pd.read_excel(filename, sheet_name=1, skiprows=None, names=columns)
    data1_2 = pd.read_excel(filename, sheet_name=2, skiprows=None, names=columns)
    data1_3 = pd.read_excel(filename, sheet_name=3, skiprows=None, names=columns)
    data1_4 = pd.read_excel(filename, sheet_name=4, skiprows=None, names=columns)
    data1_5 = pd.read_excel(filename, sheet_name=5, skiprows=None, names=columns)
    data1 = pd.concat([data1_1, data1_2, data1_3, data1_4, data1_5], axis=0)
    data1.pop('s1')  # as the data feature s1 is very small and close to 0
    data2 = data1.to_numpy(dtype=np.float32)

    return data2


# for the data augmentation of forward network
# this data augmentation method is adopted from Nat. Commun. 2023, 14, 5765
# the scale for forward network is 5 in this paper
def forward_augmentation(data, scale=5):
    data_augmented = data
    for i in range(1543):
        for j in range(4):
            for k in range(4):
                for m in range(scale):
                    if (np.argmax(data_augmented[i, 4:8]) == j) & (np.argmax(data_augmented[i, 8:12]) == k):
                        data_augmented = np.append(data_augmented, data[i, :][np.newaxis, :], axis=0)
                        tmp1 = np.random.uniform(low=0.9, high=1.0)
                        tmp1_min = np.min([tmp1, 1 - tmp1])
                        tmp2 = np.random.uniform(low=0.0, high=tmp1_min)
                        tmp2_min = np.min([tmp2, 1 - (tmp1 + tmp2)])
                        tmp3 = np.random.uniform(low=0.0, high=tmp2_min)
                        tmp4 = 1 - (tmp1 + tmp2 + tmp3)
                        tmp = [tmp2, tmp3, tmp4]
                        np.random.shuffle(tmp)
                        tmp5 = np.random.uniform(low=0.9, high=1.0)
                        tmp5_min = np.min([tmp5, 1 - tmp5])
                        tmp6 = np.random.uniform(low=0.0, high=tmp5_min)
                        tmp6_min = np.min([tmp6, 1 - (tmp5 + tmp6)])
                        tmp7 = np.random.uniform(low=0.0, high=tmp6_min)
                        tmp8 = 1 - (tmp5 + tmp6 + tmp7)
                        tmp9 = [tmp6, tmp7, tmp8]
                        np.random.shuffle(tmp9)
                        if (j == 0) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 1) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 2) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 3) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        else:
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
    # curves with one local peak and valley
    for i in range(1543, 1731):
        for j in range(4):
            for k in range(4):
                for m in range(8 * scale):
                    if (np.argmax(data_augmented[i, 4:8]) == j) & (np.argmax(data_augmented[i, 8:12]) == k):
                        data_augmented = np.append(data_augmented, data[i, :][np.newaxis, :], axis=0)
                        tmp1 = np.random.uniform(low=0.9, high=1.0)
                        tmp1_min = np.min([tmp1, 1 - tmp1])
                        tmp2 = np.random.uniform(low=0.0, high=tmp1_min)
                        tmp2_min = np.min([tmp2, 1 - (tmp1 + tmp2)])
                        tmp3 = np.random.uniform(low=0.0, high=tmp2_min)
                        tmp4 = 1 - (tmp1 + tmp2 + tmp3)
                        tmp = [tmp2, tmp3, tmp4]
                        np.random.shuffle(tmp)
                        tmp5 = np.random.uniform(low=0.9, high=1.0)
                        tmp5_min = np.min([tmp5, 1 - tmp5])
                        tmp6 = np.random.uniform(low=0.0, high=tmp5_min)
                        tmp6_min = np.min([tmp6, 1 - (tmp5 + tmp6)])
                        tmp7 = np.random.uniform(low=0.0, high=tmp6_min)
                        tmp8 = 1 - (tmp5 + tmp6 + tmp7)
                        tmp9 = [tmp6, tmp7, tmp8]
                        np.random.shuffle(tmp9)
                        if (j == 0) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 1) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 2) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 3) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        else:
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
    # curve with two local peaks and valleys
    for i in range(1731, 1870):
        for j in range(4):
            for k in range(4):
                for m in range(11 * scale):
                    if (np.argmax(data_augmented[i, 4:8]) == j) & (np.argmax(data_augmented[i, 8:12]) == k):
                        data_augmented = np.append(data_augmented, data[i, :][np.newaxis, :], axis=0)
                        tmp1 = np.random.uniform(low=0.9, high=1.0)
                        tmp1_min = np.min([tmp1, 1 - tmp1])
                        tmp2 = np.random.uniform(low=0.0, high=tmp1_min)
                        tmp2_min = np.min([tmp2, 1 - (tmp1 + tmp2)])
                        tmp3 = np.random.uniform(low=0.0, high=tmp2_min)
                        tmp4 = 1 - (tmp1 + tmp2 + tmp3)
                        tmp = [tmp2, tmp3, tmp4]
                        np.random.shuffle(tmp)
                        tmp5 = np.random.uniform(low=0.9, high=1.0)
                        tmp5_min = np.min([tmp5, 1 - tmp5])
                        tmp6 = np.random.uniform(low=0.0, high=tmp5_min)
                        tmp6_min = np.min([tmp6, 1 - (tmp5 + tmp6)])
                        tmp7 = np.random.uniform(low=0.0, high=tmp6_min)
                        tmp8 = 1 - (tmp5 + tmp6 + tmp7)
                        tmp9 = [tmp6, tmp7, tmp8]
                        np.random.shuffle(tmp9)
                        if (j == 0) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 1) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 2) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 3) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        else:
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
    # curves with three local peaks and valleys
    for i in range(1870, 1984):
        for j in range(4):
            for k in range(4):
                for m in range(14 * scale):
                    if (np.argmax(data_augmented[i, 4:8]) == j) & (np.argmax(data_augmented[i, 8:12]) == k):
                        data_augmented = np.append(data_augmented, data[i, :][np.newaxis, :], axis=0)
                        tmp1 = np.random.uniform(low=0.9, high=1.0)
                        tmp1_min = np.min([tmp1, 1 - tmp1])
                        tmp2 = np.random.uniform(low=0.0, high=tmp1_min)
                        tmp2_min = np.min([tmp2, 1 - (tmp1 + tmp2)])
                        tmp3 = np.random.uniform(low=0.0, high=tmp2_min)
                        tmp4 = 1 - (tmp1 + tmp2 + tmp3)
                        tmp = [tmp2, tmp3, tmp4]
                        np.random.shuffle(tmp)
                        tmp5 = np.random.uniform(low=0.9, high=1.0)
                        tmp5_min = np.min([tmp5, 1 - tmp5])
                        tmp6 = np.random.uniform(low=0.0, high=tmp5_min)
                        tmp6_min = np.min([tmp6, 1 - (tmp5 + tmp6)])
                        tmp7 = np.random.uniform(low=0.0, high=tmp6_min)
                        tmp8 = 1 - (tmp5 + tmp6 + tmp7)
                        tmp9 = [tmp6, tmp7, tmp8]
                        np.random.shuffle(tmp9)
                        if (j == 0) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 1) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 2) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 3) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        else:
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
# curves with four local peaks and valleys
    for i in range(1984, 2065):
        for j in range(4):
            for k in range(4):
                for m in range(19 * scale):
                    if (np.argmax(data_augmented[i, 4:8]) == j) & (np.argmax(data_augmented[i, 8:12]) == k):
                        data_augmented = np.append(data_augmented, data[i, :][np.newaxis, :], axis=0)
                        tmp1 = np.random.uniform(low=0.9, high=1.0)
                        tmp1_min = np.min([tmp1, 1 - tmp1])
                        tmp2 = np.random.uniform(low=0.0, high=tmp1_min)
                        tmp2_min = np.min([tmp2, 1 - (tmp1 + tmp2)])
                        tmp3 = np.random.uniform(low=0.0, high=tmp2_min)
                        tmp4 = 1 - (tmp1 + tmp2 + tmp3)
                        tmp = [tmp2, tmp3, tmp4]
                        np.random.shuffle(tmp)
                        tmp5 = np.random.uniform(low=0.9, high=1.0)
                        tmp5_min = np.min([tmp5, 1 - tmp5])
                        tmp6 = np.random.uniform(low=0.0, high=tmp5_min)
                        tmp6_min = np.min([tmp6, 1 - (tmp5 + tmp6)])
                        tmp7 = np.random.uniform(low=0.0, high=tmp6_min)
                        tmp8 = 1 - (tmp5 + tmp6 + tmp7)
                        tmp9 = [tmp6, tmp7, tmp8]
                        np.random.shuffle(tmp9)
                        if (j == 0) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 1) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 2) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 3) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        else:
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]

    return data_augmented


# for the data augmentation of inverse network
# The scale for inverse network is 20 in this paper
def inverse_augmentation(data, scale=20):
    data_augmented = data
    for i in range(data.shape[0]):
        for j in range(4):
            for k in range(4):
                for m in range(scale):
                    if (np.argmax(data_augmented[i, 4:8]) == j) & (np.argmax(data_augmented[i, 8:12]) == k):
                        data_augmented = np.append(data_augmented, data[i, :][np.newaxis, :], axis=0)
                        tmp1 = np.random.uniform(low=0.9, high=1.0)
                        tmp1_min = np.min([tmp1, 1 - tmp1])
                        tmp2 = np.random.uniform(low=0.0, high=tmp1_min)
                        tmp2_min = np.min([tmp2, 1 - (tmp1 + tmp2)])
                        tmp3 = np.random.uniform(low=0.0, high=tmp2_min)
                        tmp4 = 1 - (tmp1 + tmp2 + tmp3)
                        tmp = [tmp2, tmp3, tmp4]
                        np.random.shuffle(tmp)
                        tmp5 = np.random.uniform(low=0.9, high=1.0)
                        tmp5_min = np.min([tmp5, 1 - tmp5])
                        tmp6 = np.random.uniform(low=0.0, high=tmp5_min)
                        tmp6_min = np.min([tmp6, 1 - (tmp5 + tmp6)])
                        tmp7 = np.random.uniform(low=0.0, high=tmp6_min)
                        tmp8 = 1 - (tmp5 + tmp6 + tmp7)
                        tmp9 = [tmp6, tmp7, tmp8]
                        np.random.shuffle(tmp9)
                        if (j == 0) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 0) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 5] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 1) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 1) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 6] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 2) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 2) & (k == 3):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 7] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]
                        elif (j == 3) & (k == 0):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 9] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 1):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 10] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        elif (j == 3) & (k == 2):
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 11] = tmp9[2]
                        else:
                            data_augmented[-1, 4 + j] = tmp1
                            data_augmented[-1, 4] = tmp[0]
                            data_augmented[-1, 5] = tmp[1]
                            data_augmented[-1, 6] = tmp[2]
                            data_augmented[-1, 8 + k] = tmp5
                            data_augmented[-1, 8] = tmp9[0]
                            data_augmented[-1, 9] = tmp9[1]
                            data_augmented[-1, 10] = tmp9[2]

    return data_augmented


# define the loss function and metrics mean absolute error (MAE)
def loss_function(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def loss_mae(y_true, y_pred):
    return float(K.mean(K.abs(y_true - y_pred)))
