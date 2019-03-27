#coding=utf-8

import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np

prams = {
'm':5,
'learning_rate':0.01,
'epochs':1000
}

def load_data(data):
    """
    :param data:
    :return:
    """
    return X_train,y_train,X_test,y_test

class MLR(object):
    def __init__(self,feature_size,
                 m=5,
                 Lambda=0.2,
                 beta =0.6,
                 learn_rate=0.01,
                 epochs=1000):
        self.feature_size = feature_size  #dim of data
        self.m = m    #tilce
        self.Lambda = Lambda
        self.beta = beta
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._init_graph()

    def _init_graph(self):

        self.x = tf.placehoder(tf.float32,shape=[None,self.feature_size])
        self.label = tf.placehoder(tf.float32,shape=[None])

        self.u = tf.Variable(tf.random_normal([self.feature_size,self.m],0.0,0.05),name='u')
        self.w = tf.Variable(tf.random_normal([self.feature_size,self.m],0.0,0.05),name='w')

        self.U = tf.mutmul(x,self.u)
        self.p1 = tf.nn.softmax(self.U)

        self.W = tf.mutmul(x,self.w)
        self.p2 = tf.nn.sigmoid(self.W)

        self.pred = tf.reduce_sum(tf.multiply(self.p1,self.p2),1)
        l1 = 0
        l21 = 0
        self.square_u = tf.square(self.u)
        self.square_w = tf.square(self.w)

        l1 = tf.reduce_sum(tf.abs(tf.concat([self.u,self.w],1)))
        l21 = tf.reduce_sum(
            tf.sqrt(
                tf.reduce_sum(tf.concat([tf.square(u),tf.square(w)],1))
            )
        )
        self.regularization = self.Lambda*l21 + self.beta *l1
        self.loss = tf.add_n([tf.reduce_sum(tf.nn.sigmoid_entropy_with_logits(
            logits=pred,labels=label))])
        self.loss = self.loss+self.regularization
        self.optimizer = tf.train.FtrlOptimizer(learning_rate).minimize(loss)

        # init
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def train(self,X_train,y_train,X_test,y_test):

        for epoch in range(self.epochs):
            feed_dict = {x:X_train,y:y_train}
            _,loss,pred = self.sess.run([self.optimizer,self.loss,self.pred],feed_dict=feed_dict)
            if epoch %100 ==0:
                feed_dict = {x:X_test,y:y_test}
                _,loss,pred = self.sess.run([self.optimizer,self.loss,self,pred],feed_dict=feed_dict)
                auc = roc_auc_score(y_test,pred)
                print 'epoch:',epoch,'auc:',auc

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data(data)
    prams['feature_size'] = X_train.shape[1]
    mlr = MLR(**prams)
    mlr.train(X_train, y_train, X_test, y_test)