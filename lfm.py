#coding=utf-8
# @author@:zxf

import math
import numpy as np

def LFM(A,k,learning_rate,lambda_value):
    """

    :param A: 原始矩阵,adarray
    :param k: 隐变量的个数
    :param learning_rate: 学习率
    :param lambda_value: 正则化系数
    :return:
    """
    assert type(A) == np.ndarray
    m,n = A.shape
    u = np.random.rand(m,k)
    v = np.random.randn(k,n)
    for step in range(1000):
        for i in range(m):
            for j in range(n):
                if math.fabs(A[i][j])>1e-4:
                    error = A[i][j]-np.dot(u[i],v[:,j])
                    for q in range(k):
                        # 梯度下降求解
                        loss_u = error*v[q][j]-lambda_value*u[i][q]
                        loss_v = error*u[i][q]-lambda_value*v[q][j]
                        u[i][q] += learning_rate*loss_u
                        v[q][j] += learning_rate*loss_v
    return u,v


if __name__ == '__main__':
    A = np.array([[4,2,0,4,5],[5,3,1,5,2],
                  [1,4,2,0,1],[5,3,1,0,1],[5,2,5,3,0]])
    u,v = LFM(A,3,0.01,0.01)
    a = np.dot(u,v)
    print a



