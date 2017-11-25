###################################################################################
#   coding =utf-8
#   author: jinzhong xu
#   对于给定的空间点集计算集合的单纯复形的滤子熵
###################################################################################

import math
import matplotlib.pyplot as plt
import numpy as np
import DeduceDup


def ComputeEntropy(x, M):
    '''
    用以计算数据集的熵，并画出熵图
    :param x: 数据集x，以列表形式表示，每个元素为元组形式
    :param M: 参数M用以设置熵图最右侧边距
    :return: 得到数据的距离向量，熵向量和熵图
    '''
    #    M = 10  # 设置熵图最右侧边距
    DM = np.zeros((len(x), len(x)))    #初始化距离矩阵，然后计算距离矩阵
    for i in np.arange(0, len(x)):
        for j in np.arange(i + 1, len(x)):
#            for k in np.arange(0, len(x[0])):
#                DM[i][j] = DM[i][j] + (x[i][k] - x[j][k]) ** 2   #这里采用欧式距离
#            DM[i][j] = math.sqrt(DM[i][j])
            x[i] = np.array(x[i])
            x[j] = np.array(x[j])
            DM[i][j] = np.linalg.norm(x[i] - x[j], ord = 2)    #可以规定不同的范数ord
    print("DistanceMatrix:\n{0}".format(DM))    # test    打印距离矩阵
    DMv = DM.reshape((1, len(x) * len(x)))    # 把距离矩阵转化为向量形式
    dmv = np.nonzero(DMv[0])    # 索引距离矩阵向量化后的非零位置
    dm = DMv[0][dmv]    # 把非零数字提取出来，得到距离向量
    #   print(type(dm))             #此处dm是数组
    dm = sorted(dm)    #得到排好序的距离向量
    #   print(type(dm))             #此处dm是列表list
    dm = np.array(dm)     #把列表转化为数组，以便调用函数DeduceDup.dedup(d)
    dm = DeduceDup.dedup(dm)      #把距离向量中重复出现的值只保留一个，此处dm格式为一维数组
    print("DistanceMatrixConvertToVector: {0}".format(dm))  # test      打印距离向量


    circle = [1] * len(x)  # 用于存储x_0,...,x_n的圆邻域内包含的点数，初始化时每个圆内有一个点，就是圆心本身
    h = [0] * (len(dm) + 1)  # 初始化熵向量
    hdm = [0] * (len(dm) + 1)  # 初始化用于画数图的熵向量
    for k in np.arange(1, len(h) - 1):
        for i in np.arange(0, len(circle)):
            for j in np.arange(len(x)):  # 内层两个for循环用于计算圆邻域内的点数
                for t in np.arange(j + 1, len(x)):
                    if DM[j][t] <= dm[k - 1]:
                        circle[j] += 1
                        circle[t] += 1
            h[k] = h[k] - (circle[i] / sum(circle)) * math.log(circle[i] / sum(circle))  # 计算原始熵，归一化之前的值
            if i == len(circle) - 1:
                print("NumbersOfPointsInCircle: {0} when distance is {1}".format(circle, dm[k - 1]))  # 打印圆邻域向量
            circle = [1] * len(x)  # normalized 再次进行归一化圆邻域向量
        e2 = np.arange(dm[k - 1], dm[k], 0.000001)  # 离散化区间（dm[k - 1], dm[k]）用以画图
        h[k] = h[k] / math.log(len(x))  # 归一化熵
        hdm[k] = h[k] * np.ones(len(e2))  # 离散化熵用以画图
        plt.plot(e2, hdm[k], color = "black")  # 画出距离为dm[k - 1]->dm[k]的熵图

    e1 = np.arange(0, dm[0], 0.000001)  # 距离从0到最短两点距离时，归一化后的熵值都为1
    h[0] = (math.log(len(x)) / math.log(len(x)))
    hdm[0] = h[0] * np.ones(len(e1))
    plt.plot(e1, hdm[0], color = "black")  # 画出第一段熵图

    e3 = np.arange(dm[len(dm) - 1], dm[len(dm) - 1] + M, 0.000001)  # M为边距值，用以修补无穷区间，而画出有限区间，这是最后一段的值恒定为1
    h[-1] = (math.log(len(x)) / math.log(len(x)))
    hdm[-1] = h[-1] * np.ones(len(e3))
    plt.plot(e3, hdm[-1], color = "black")  # 画出最后一段熵图
    
#    plt.xticks(np.linspace(0, dm[len(dm) - 1] + M, 11))  #  使坐标轴更精细
#    plt.yticks(np.linspace(0.90, 1.05, 16))                    
    plt.xlim([0, dm[len(dm) - 1] + M])
    plt.ylim([0.90, 1.05])  # 格式化坐标轴显示区间
    print("ComplexEntropyVector:{0}".format(h))
    print("Min(h)=", min(h))
    print("Max(h)=", max(h))
    print("Ave(h)=", sum(h) / len(h))
    #   print(type(h))                  #h输出是列表list格式
    plt.title("$Filtration\ Entropy\ of\ Simplicial\ Complexes(FESC)$")
    plt.xlabel("$Distance\ Parameter(or\ time): \epsilon$")
    plt.ylabel("$Filtration\ Entropy: h_X(\epsilon)$")
    plt.show()

    #   内部执行本函数需下面代码
    # if __name__ == "__main__":
    #    ComputeEntropy(x)
