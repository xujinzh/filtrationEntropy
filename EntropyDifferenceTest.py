##################
#   coding =utf-8
#   author: jinzhong xu
#   载入数据集，计算数据熵和熵亏
##################

import numpy as np
import ComputeDifferenceEntropy

d1 = [1.0, 2.0, 2.23606798]
d2 = [1.0, 2.06155281]
d1 = np.array(d1)
d2 = np.array(d2)
h1 = [1.0, 0.9602297178607612, 0.9821410328348751, 1.0]
h2 = [1.0, 0.9602297178607612, 1.0]

ComputeDifferenceEntropy.ComputeDifferEntopy(d1, d2, h1, h2)
