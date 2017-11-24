##################
#   coding =utf-8
#   author: jinzhong xu
#   计算两个数据集合的数据熵亏
##################

import numpy as np
import math
import DeduceDup


def ComputeDifferEntopy(d1, d2, h1, h2):
    '''
    计算两数据集的相对熵亏值
    :param d1: 数据1距离向量，格式为一维数组
    :param d2: 数据2距离向量，格式为一维数组
    :param h1: 数据1对于距离向量d1的熵值，格式为列表
    :param h2: 数据2对于距离向量d2的熵值，格式为列表
    :return: 返回数据1和数据2的相对数据熵亏值
    '''
    d = list(d1) + list(d2)
    d = sorted(d)
    d = np.array(d)
    d = DeduceDup.dedup(d)
    print("d=", d)  # test
    print("d1=", d1)  # test
    print("d2=", d2)  # test

    dmin = max(d1[0], d2[0])
    print("d[i_min]=", dmin)  # test
    dmax = min(d1[-1], d2[-1])
    print("d[i_max]=", dmax)  # test
    for i in np.arange(len(d)):
        if dmin == d[i]:
            i_min = i
        if dmax == d[i]:
            i_max = i
    print("i_min=", i_min, "i_max=", i_max)  # test
    # 计算中间部分熵亏值
    h_middle = 0
    hx, hy = 0, 0
    dx, dy = 0, 0
    for i in np.arange(i_min, i_max):
        for j in np.arange(len(d1) - 1):
            if d1[j] <= d[i] < d1[j + 1]:
                hx = h1[j + 1]
                dx = min(d1[j + 1] - d1[j], d[i_max] - d1[j])
            else:
                hx = 0
                dx = 0
        for k in np.arange(len(d2) - 1):
            if d2[k] <= d[i] < d2[k + 1]:
                hy = h2[k + 1]
                dy = min(d2[k + 1] - d2[k], d[i_max] - d2[k])
            else:
                hy = 0
                dy = 0
        h_middle = h_middle + math.fabs(hx - hy) * min(dx, dy)

    print("h_middle=", h_middle)  # test
    # 计算左半部分熵亏值
    h_zuoban, h_qian = 0, 0
    if d[i_min] == d1[0]:
        for i in np.arange(len(d2) - 1):
            if d2[i] <= d[i_min] < d2[i + 1]:
                hy = h2[i + 1]
                dy = d[i_min] - d2[i]
                h_ban = (1 - hy) * dy
                for j in np.arange(0, i):
                    h_qian = h_qian + (1 - h2[j + 1]) * (d2[j + 1] - d2[j])
        h_qian = h_qian + h_zuoban
    if d[i_min] == d2[0]:
        for i in np.arange(len(d1) - 1):
            if d1[i] <= d[i_min] < d1[i + 1]:
                hx = h1[i + 1]
                dx = d[i_min] - d1[i]
                h_zuoban = (1 - hx) * dx
                for j in np.arange(0, i):
                    h_qian = h_qian + (1 - h1[j + 1]) * (d1[j + 1] - d1[j])
        h_qian = h_qian + h_zuoban

    print("h_qian=", h_qian)  # test
    # 计算有半部分熵亏值
    h_youban, h_hou = 0, 0
    if d[i_max] == d1[-1]:
        for i in np.arange(len(d2) - 1):
            if d2[i] <= d[i_max] < d2[i + 1]:
                hy = h2[i + 1]
                dy = d2[i + 1] - d[i_max]
                h_youban = (1 - hy) * dy
                for j in np.arange(i + 1, len(d2) - 1):
                    h_hou = h_hou + (1 - h2[j + 1]) * (d2[j + 1] - d2[j])
        h_hou = h_hou + h_youban
    if d[i_max] == d2[-1]:
        for i in np.arange(len(d1) - 1):
            if d1[i] <= d[i_max] < d1[i + 1]:
                hx = h1[i + 1]
                dx = d1[i + 1] - d[i_max]
                h_youban = (1 - hx) * dx
                for j in np.arange(i + 1, len(d1) - 1):
                    h_hou = h_hou + (1 - h1[j + 1]) * (d1[j + 1] - d1[j])
        h_hou = h_hou + h_youban

    print("h_hou=", h_hou)  # test

    h = h_qian + h_middle + h_hou  # 计算整个部分的熵亏值
    print("DataEntropyDifference Dh_r(X,Y) = ", h)
