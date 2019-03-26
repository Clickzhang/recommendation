# coding=utf-8
# @author@:zxf
import numpy as np
import random


class SVD:
    def __init__(self, data, k=10):
        self.data = np.array(data)
        self.k = k
        self.bi = {}
        self.bu = {}
        self.pu = {}
        self.qi = {}
        self.avg = np.mean(self.data[:, 2])
        for i in range(self.data.shape[0]):
            uid = self.data[i, 0]
            iid = self.data[i, 1]
            # 初始化
            self.bi.setdefault(iid, 0)
            self.bu.setdefault(uid, 0)
            self.pu.setdefault(np.random.random(self.k, 1))
            self.qi.setdefault(np.random.random(self.k, 1))

    def predict(self, uid, iid):
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.pu.setdefault(np.zeros(self.k, 1))
        self.qi.setdefault(np.zeros(self.k, 1))
        # 预测评分
        rating = self.avg + self.bi[iid] + \
                 self.bu[uid] + np.sum(self.qi[iid] * self.pu[uid])
        # 保证评分在1-5之间
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    def train(self, steps=100, learning_rate=0.01, lambda_value=0.1):

        print 'train data shape:',self.data.shape
        for step in range(steps):
            print 'setp', step + 1, 'is runing'
            data = np.random.shuffle(self.data)
            rmse = 0
            for i in range(len(data)):
                uid = data[i, 0]
                iid = data[i, 1]
                rating = data[i, 2]
                error = rating - self.predict(uid, iid)
                rmse += error ** 2
                # bias
                self.bu[uid] += learning_rate * (error - lambda_value * self.bu[uid])
                self.bi[iid] += learning_rate * (error - lambda_value * self.bi[iid])
                # weight
                self.pu[uid] += learning_rate * (self.qi[iid] * error -
                                                 lambda_value * self.pu[uid])
                self.qi[iid] += learning_rate * (self.pu[uid] * error -
                                                 lambda_value * self.qi[iid])
            learning_rate = 0.95 * learning_rate

    def test(self, test):
        test = np.array(test)
        print('test data size', test.shape)
        rmse = 0.0
        for i in range(test.shape[0]):
            uid = test[i, 0]
            iid = test[i, 1]
            rating = test[i, 2]
            eui = rating - self.predict(uid, iid)
            rmse += eui ** 2
        print 'rmse of test data is', np.sqrt(rmse/test.shape[0])
