#coding=utf-8
import tensorflow as tf
import random
import numpy as np

class FM(object):
    def __init__(self,
                 feature_size,
                 embedding_size=6,
                 lambda_w=0.1,
                 lambda_v=0.1,
                 batch_size=256,
                 epochs = 10,
                 learning_rate=0.01):
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.lambda_w = lambda_w
        self.lambda_v = lambda_v
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self._init_graph()

    def _init_graph(self):

        x = tf.placehoder(tf.float32,shape=[None,self.feature_size])
        y = tf.placehoder(tf.float32,shape=[None,1])

        self.w0 = tf.Variable(tf.zeros([1]))
        self.w = tf.Variable(tf.zeros([self.feature_size]))

        self.v = tf.Variable(
            tf.random_normal([self.feature_size,self.embedding_size],0.0,0.05))

        linear_terms = tf.add(self.w0,tf.reduce_sum(tf.multiply(self.w,x),1,keepdims=True))  #n*1
        interaction_terms = 0.5*tf.reduce_sum(
            tf.substract(
                tf.pow(tf.matmul(x,self.v),2),tf.matmul(tf.pow(x,2),tf.pow(self.v,2))
            ),1,keepdoms=True)
        self.y_hat = tf.add(linear_terms,interaction_terms)

        l2_norm = tf.reduce_sum(
            tf.add(
                tf.multiply(self.lambda_w,tf.pow(w,2)),
                tf.multiply(self.lambda_v,tf.pow(v,2)))
        )
        self.loss = tf.add(tf.reduce_sum(tf.square(y-y_hat)),l2_norm)

        self.optimizer = tf.train.GradientDescent(
            learning_rate=self.learning_rate).minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def get_batch(self,X_train,y_train):

        batch_index = np.random.shuffle(range(len(X_train)))[:self.batch_size]
        return X_train[batch_index],y_train[batch_index]

    def train(self,X_train,y_train,X_valid,y_valid):

        LOSS = []
        for epoch in range(self.epochs):
            batch_X,batch_y = self.get_batch(X_train,y_train)
            _,loss = tf.sess.run([self.optimizer,self.loss],
                                 feed_dict={x:batch_X.reshpae(-1,self.feature_size),
                                            y:batch_y.reshape(-1,1)})
            LOSS.append(loss)
        print 'RMSE',np.sqrt(np.array(LOSS).mean())






