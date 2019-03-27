# coding=utf-8
import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NFM(BaseEstimator, TransformerMixin):
    def __init__(self,
                 feature_size,
                 field_size,
                 embedding_size=8,
                 deep_layers=[32, 32],
                 deep_init_size=50,
                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layer_activation=tf.nn.relu,
                 epoch=10,
                 batch_size=256,
                 learning_rate=0.005,
                 optimizer_type="adam",
                 verbose=False,
                 random_seed=2019,
                 loss_type="logloss",
                 eval_metric=roc_auc_score,
                 # know the means
                 batch_norm=0,
                 batch_norm_decay=0.995,
                 greater_is_better=True,
                 use_inner=True):
        assert loss_type in ["logloss", 'mse'], "just for logloss or mse"
        self.feature_size = feature_size
        self.field_size = field_size  # F
        self.embedding_size = embedding_size  # K

        self.deep_layers = deep_layers
        self.deep_init_size = deep_init_size
        self.dropout_deep = dropout_deep
        self.deep_layer_activation = deep_layer_activation

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result = []
        self.vaild_result = []
        self.use_inner = use_inner

        self._init_graph()

    def _init_weights(self):

        weights = {}
        # embedding
        weights['feature_embedding'] = tf.Variable(
            tf.random_normal([self.feature_size, self.embedding_size], 0.0, 1),
            name='cross_weight')
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_size, 1], 0.0, 1),
                                              name='linear_weight')
        weights['bias'] = tf.Variable(tf.constant(0.1), name='bias')

        # deep_layers
        num_layer = len(self.deep_layers)  # 网络层数
        input_size = self.embedding_size  # bi-interaction操作
        # 初始化参数分布
        if self.deep_layer_activation == tf.nn.tanh:
            glorot = np.sqrt(1.0 / input_size)
        elif self.deep_layer_activation == tf.nn.relu:
            glorot = np.sqrt(2.0 / input_size)
        else:
            glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))

        weights['layer_0'] = tf.Variable(
            np.random.normal(
                loc=0, scale=glorot, size=(input_size, self.deep_layers[0])),
            dtype=np.float32)  # input_size * layer[0]
        weights['bias_0'] = tf.Variable(
            np.random.normal(
                loc=0, scale=glorot, size=(1, self.deep_layers[0])),
            dtype=np.float32)  # 1*layer[0]

        for i in range(1, num_layer):
            # 神经元每层权值方差初始化glorot
            # 初始化参数分布
            if self.deep_layer_activation == tf.nn.tanh:
                glorot = np.sqrt(1.0 / self.deep_layers[i - 1])
            elif self.deep_layer_activation == tf.nn.relu:
                glorot = np.sqrt(2.0 / self.deep_layers[i - 1])
            else:
                glorot = np.sqrt(2.0 / (self.deep_layers[i - 1], self.deep_layers[i]))

            weights['layer_%d' % i] = tf.Variable(
                np.random.normal(
                    loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layer[i-1] * layer[i]

            weights['bias_%d' % i] = tf.Variable(
                np.random.normal(
                    loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        return weights

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)  # 初始化种子
            self.feat_index = tf.placeholder(
                tf.int32, shape=[None, None], name="feat_index")
            self.feat_value = tf.placeholder(
                tf.float32, shape=[None, None], name="feat_value")

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
            # -------------------------------------------------------------------
            # 每一层的dropout概率
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None],
                                                    name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            # ------------------------------------------------------------------
            self.weights = self._init_weights()

            # Embedding
            self.embedding = tf.nn.embedding_lookup(
                self.weights["feature_embedding"], self.feat_index)  # N*F*K(F is field_size )
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embedding = tf.multiply(self.embedding, feat_value)  # N*F*K

            # first_order_term
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_boas'], self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)

            # secend_order_term
            # sum_square_part
            self.sum_feature_emb = tf.reduce_sum(self.embedding, 1)  # None*K
            self.sum_feature_emb_square = tf.square(self.sum_feature_emb)  # None*K

            # square_sun_part
            self.square_feature_emb = tf.square(self.embedding)
            self.square_feature_emb_sum = tf.reduce_sum(self.square_feature_emb, 1)  # None*K

            # sencond order
            self.y_second_order = 0.5 * tf.subtract(self.sum_feature_emb_square, self.square_feature_emb_sum)

            # Deep component
            self.y_deep = self.y_second_order
            for i in range(len(self.deep_layers)):
                self.y_deep = tf.add(
                    tf.matmul(
                        self.y_deep, self.weights['layer_%d' % i]), self.weights['bias_%d' % i])
                self.y_deep = self.deep_layer_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])

            self.y_bias = self.weights['bias'] * tf.ones_like(self.label)


            # out
            self.out = tf.add_n([tf.reduce_sum(self.y_first_order, 1, keepdims=True),
                                 tf.reduce_sum(self.y_deep, 1, keepdims=True),
                                 self.y_bias])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)  # logloss损失函数
            if self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.substract(self.label, self.out))

            # optimizer
            if self.optimizer_type == 'gd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            if self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            if self.optimizer_type == "adadelta":
                self.optimizer = tf.train.AdadetlaOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            if self.optimizer_type == "rmsprop":
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            if self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            if self.optimizer_type == "ftrl":
                self.optimizer = tf.train.FtrlOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def get_batch(self, Xi, Xv, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)

        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    def predict(self, Xi, Xv, y):
        """
        :param Xi:
        :param XV:
        :param y:
        :return: prob of sample
        """
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_deep),  # 测试不需要dropout
                     self.train_phase: True
                     }
        prob, loss = self.sess.run([self.out, self.loss], feed_dict=feed_dict)
        return prob

    def fit_on_batch(self, Xi, Xv, y):

        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True
                     }
        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def fit_predict(self, Xi_train, Xv_train, y_train,
                    Xi_valid=None, Xv_valid=None, y_valid=None,
                    early_stoping=False, refit=False):

        has_valid = Xv_valid is not None
        for epoch in range(self.epoch):

            num_batch = int(len(y_train) / self.batch_size)
            sum_loss = 0
            for i in range(num_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                self.fit_on_batch(Xi_batch, Xv_batch, y_batch)

                if has_valid:
                    y_valid = np.array(y_valid).reshape((-1, 1))
                    prob, loss = self.predict(Xi_valid, Xv_valid, y_valid)
                    sum_loss += loss

            loss = sum_loss / num_batch
            print "epoch:", epoch, "loss:", loss
