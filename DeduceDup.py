##################
#   coding =utf-8
#   author: jinzhong xu
#   约简一维数组中的非零重复值，只保留一个
##################

import numpy as np


def dedup(d):
    '''
    剔除距离向量中的重复值
    :param d: 距离向量，一般包含重复元素值，格式为一维数组
    :return: 返回一个没有重复值的向量，格式为一维数组
    '''
    for i in np.arange(len(d) - 1):
        for j in np.arange(i + 1, len(d)):
            if d[i] >= d[j]:
                d[j] = 0
    d0_index = np.nonzero(d)
    dn = d[d0_index]
    return dn