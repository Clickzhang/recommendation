#coding=utf-8

class FFM(object):
    def __init__(self,
                 field_size,
                 feature_size,
                 embedding_size=6,
                 lambda_w = 0.01,
                 lambda_v = 0.01,
                 learning_rate = 0.01,
                 batch_size = 1000):
        self.field_size = field_size
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.lambda_w = lambda_w
        self.lambda_v = lambda_v
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self._init_graph()

    def _init_weight(self):

        weights = {}
        # embedding
        weights['interaction_embedding'] = tf.Variable(
            tf.truncated_normal([self.feature_size,self.field_size,self.embedding_size],0,0.01))
        # linear weights
        weights['linear_embedding'] = tf.Variable([self.feature_size],0,0.01)
        # bias
        weights['bias'] = tf.Variable(tf.zeros([1]))

        return weights

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.x = tf.placehoder(tf.float32,[None,self.feature_size],name='input_x')
            self.y = tf.placehoder(tf.float32,[None,1],name='label')
            self.weights = self._init_weight()

            #线性
            self.bias = self.weights['bias'] * tf.ones_like(self.y)
            self.linear_term = tf.reduce_sum(
                tf.multiply(self.weights['linear_embedding'],self.x),1,keepdims=True)
            self.first_term = tf.add(self.bias,self.linear_term,name='firstTerm')
            #交叉
            self.secend_term = tf.Variable(0,dtype=tf.float32)
            for i in range(self.feature_size):
                featureIndex1 = i
                fieldIndex1 = int(x_field)  #x_field表示feature-field对应表
                for j in range(i+1,self.feature_size):
                    featureIndex2 = j
                    fieldIndex2 = int(x_field)
                    left_embedding = tf.convert_to_tensor(
                        [featureIndex1,featureIndex2,i] for i in range(self.embedding_size))
                    left_weight = tf.gather_nd(self.weights['interaction_embedding'],left_embedding)
                    left_weight_cut = tf.squeeze(left_weight)

                    right_embedding = tf.convert_to_tensor(
                        [featureIndex2, featureIndex1, i] for i in range(self.embedding_size))
                    right_weight = tf.gather_nd(self.weights['interaction_embedding'], right_embedding)
                    right_weight_cut = tf.squeeze(right_weight)

                    sumLatent = tf.reduce_sum(tf.multiply(left_weight_cut,right_weight_cut))

                    xi = tf.squeeze(tf.gather_nd(self.x,[i]))
                    xj = tf.squeeze(tf.gather_nd(self.x,[j]))

                    sumValue = tf.reduce_sum(tf.multiply(xi,xj))

                    self.secend_term+= tf.multiply(sumLatent,sumValue)

            self.out = tf.add(self.first_term,self.secend_term)

            l2_norm = tf.reduce_sum(
                tf.add(tf.multiply(self.lambda_w,tf.pow(self.weights['linear_embedding'],2)),
                tf.reduce_sum(tf.multiply(
                    self.lambda_v,tf.pow(self.weights['interaction_embedding'],2)),axis=[1,2])
                )
            )
            self.loss = 1 + tf.log(1+tf.exp(self.y*self.out)) + l2_norm

            self.optimizer = tf.train.GradientDiscent(learning_rate=self.learning_rate).minimize(self.loss)

            #init
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.sun(init)

    #一次训练一个样本
    def train(self,X_train,y_train,X_valid,y_valid):

        for i in range(len(X_train)):
            input_x = X_train[i]
            input_y = y_train[i]
            feed_dict = {self.x:input_x,
                         self.y:input_y
            }
            _,loss = self.sess.run([self.optimizer,self.loss],feed_dict=feed_dict)













