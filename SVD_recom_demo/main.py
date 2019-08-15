# -*- coding: utf-8 -*-
import numpy as np
import time
import pickle
'''
     可以参考知乎这里的说明：https://zhuanlan.zhihu.com/p/42147194
       感觉这里的SVD相当于是LFM的简化版吧
'''
class SVD(object):
    """
    implementation of SVD for CF “https://zhuanlan.zhihu.com/p/42147194/”
    Reference:
    A Guide to Singular Value Decomposition for Collaborative Filtering

    核心是:SVD在进行用户i对商品j的评分时，考虑了用户特征向量和商品特征向量，以及评分相对于平均评分的偏置向量。

    发现跟 LFM算法基本是一致的，核心就是下面的式子， 也是通过一定的计算来模拟构建 兴趣矩阵，只不过这里构建矩阵时，要考虑的参数
    包括 U 矩阵参数、M矩阵参数 、bu 、bi 这几种参数，其他都是差不多，  其余就是通过构建兴趣预估后的训练过程，这里正负样例是不需要构建的，
    因为用户的评分值rate是已知的，可以当做标签来使用。

    其实本质上是找到 用户特征矩阵U 、 物品特征向量M,学会对这两个向量的模拟。  这里新加的bi和bu整好应对之前感觉使用坐标来定位感觉信息不够，
    bi和bu是新加的信息，其中bi表示电影i的评分相对于平均评分的偏差，bu表示用户u所做的评分相对于平均评分的偏差，相当于是一种补足信息吧。

        p = self.meanV + self.bu[uid] + self.bi[iid] + np.sum(self.U[uid] * self.M[iid])
    """

    def __init__(self, epoch, eta, userNums, itemNums, ku=0.001, km=0.001,f=30, save_model=False):
        super(SVD, self).__init__()
        self.epoch = epoch
        self.userNums = userNums
        self.itemNums = itemNums
        self.eta = eta
        self.ku = ku
        self.km = km
        self.f = f
        self.save_model = save_model

        self.U = None
        self.M = None

    def fit(self, train, val=None):
        rateNums = train.shape[0]
        self.meanV = np.sum(train[:, 2]) / rateNums
        initv = np.sqrt((self.meanV - 1) / self.f)
        self.U = initv + np.random.uniform(-0.01, 0.01, (self.userNums + 1, self.f))
        self.M = initv + np.random.uniform(-0.01, 0.01, (self.itemNums + 1, self.f))
        self.bu = np.zeros(self.userNums + 1)
        self.bi = np.zeros(self.itemNums + 1)

        start = time.time()
        for i in range(self.epoch):
            sumRmse = 0.0
            for sample in train:
                uid = sample[0]
                iid = sample[1]
                vij = float(sample[2])
                # p(U_i,M_j) = mu + b_i + b_u + U_i^TM_j
                p = self.meanV + self.bu[uid] + self.bi[iid] + \
                    np.sum(self.U[uid] * self.M[iid])
                error = vij - p
                sumRmse += error ** 2
                # 计算Ui,Mj的梯度
                deltaU = error * self.M[iid] - self.ku * self.U[uid]
                deltaM = error * self.U[uid] - self.km * self.M[iid]
                # 更新参数
                self.U[uid] += self.eta * deltaU
                self.M[iid] += self.eta * deltaM

                self.bu[uid] += self.eta * (error - self.ku * self.bu[uid])
                self.bi[iid] += self.eta * (error - self.km * self.bi[iid])

            trainRmse = np.sqrt(sumRmse / rateNums)

            if val.any():
                _, valRmse = self.evaluate(val)
                print("Epoch %d cost time %.4f, train RMSE: %.4f, validation RMSE: %.4f" % \
                      (i, time.time() - start, trainRmse, valRmse))
            else:
                print("Epoch %d cost time %.4f, train RMSE: %.4f" % \
                      (i, time.time() - start, trainRmse))

        if self.save_model:
            save_model='../data'
            model = (self.meanV, self.bu, self.bi, self.U, self.M)
            pickle.dump(model, open(save_model + '/svcRecModel.pkl', 'wb'))

    def evaluate(self, val):

        '''
        根据用户id和商品 id  去定位 去按照预估的参数，计算出定位得分，得到兴趣度评估分。
        '''
        loss = 0
        pred = []
        for sample in val:
            uid = sample[0]
            iid = sample[1]
            if uid > self.userNums or iid > self.itemNums:
                continue

            predi = self.meanV + self.bu[uid] + self.bi[iid] \
                    + np.sum(self.U[uid] * self.M[iid])
            if predi < 1:
                predi = 1
            elif predi > 5:
                predi = 5
            pred.append(predi)

            if val.shape[1] == 3:
                vij = sample[2]
                loss += (predi - vij) ** 2

        if val.shape[1] == 3:
            rmse = np.sqrt(loss / val.shape[0])
            return pred, rmse

        return pred

    def predict(self, test):

        return self.evaluate(test)


def test():
    import pandas as pd
    data_path = '../data/ml-1m/ratings.dat'
    data = pd.read_csv(data_path, sep='::', header=None,names=['user', 'item', 'rate', 'time'], engine='python')

    data = data.sample(frac=1)
    print(data.head())

    del data['time']
    trainNum = int(data.shape[0] * 0.8)
    train = data[:trainNum].values
    val = data[trainNum:].values

    userNums = data['user'].max()
    itemNums = data['item'].max()
    svd = SVD(35, 0.001, userNums, itemNums, f=50)
    svd.fit(train, val=val)
    svd.predict(val)
if __name__ == '__main__':
    test()
