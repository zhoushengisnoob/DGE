import tensorflow as tf
import math
import random
import numpy as np
import sys
import scipy.io as sio
sys.path.append('..')
print(sys.path)
from util.evaluation import eval_classification, eval_link_prediction
from util import load_data

dataset = 'citeseer'
prefix = '../datasets/'+dataset+'/'+dataset


def get_all_data(feature_file, label_file, edge_file):
    print('read data')
    feature, N, att_dim = load_data.read_feature(feature_file)
    label = load_data.read_label(label_file)
    A, adj, edge_dict = load_data.read_edgelist(edge_file, N)
    # A sum(A[0]) = 1
    return feature, A, label, adj, edge_dict, N, att_dim


class VAE:
    def __init__(self, N, att_dim, batch_size, struct, A, X, network, label, test_lp_data=None):
        self.N = N
        self.att_dim = att_dim
        self.batch_size = batch_size
        self.test_lp_data = test_lp_data
        self.struct = struct
        self.X = tf.Variable(X, dtype=tf.float32, trainable=False)
        self.A = tf.Variable(A, dtype=tf.float32, trainable=False)
        # self.edge_dict = edge_dict
        self.label = label
        self.K = len(set(label))
        self.network = network
        self.activation = tf.nn.sigmoid
        self.alpha = 0.8
        early_stopping_score_max = -1.0
        # placeholder
        self.input = tf.placeholder(tf.int32, [self.N], name='input1')
        self.att_input = tf.nn.embedding_lookup(self.X, self.input)
        self.net_input = tf.nn.embedding_lookup(self.A, self.input)

        self.encoder_mu, self.encoder_log_var = self.encoder(self.att_input, self.net_input)
        self.z = self.reparameterization_trick(self.encoder_mu, self.encoder_log_var)
        net_logit = self.net_decoder(self.z)

        att_logit = self.att_decoder_bernoulli(self.z)
        # loss
        self.kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + self.encoder_log_var - tf.square(self.encoder_mu) - tf.exp(self.encoder_log_var), axis=1))
        self.net_loss = self.get_net_loss(net_logit)
        # self.att_loss = self.get_att_loss_bernoulli(att_logit, self.X)
        self.att_loss = self.get_att_loss_gaussian(att_logit, self.X)
        self.reg_loss = tf.losses.get_regularization_loss()

        self.loss = self.net_loss + self.att_loss + self.kl_loss + self.reg_loss
        self.optimizer = tf.train.AdamOptimizer()
        # grads = self.optimizer.compute_gradients(self.loss)
        # grads_var = [v for (g, v) in grads if g is not None]
        # grads_new = self.optimizer.compute_gradients(self.loss, grads_var)
        # self.train_op = self.optimizer.apply_gradients([(tf.clip_by_norm(gv[0], 10), gv[1]) for gv in grads_new])
        self.train_op = self.optimizer.minimize(self.loss)

        # writer = tf.summary.FileWriter('path/to/log/', tf.get_default_graph())
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def encoder(self, att_input_, net_input_):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            att_layer = tf.layers.dense(att_input_, self.struct['encoder_att_hidden1'], activation=self.activation,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.alpha))
            net_layer = tf.layers.dense(net_input_, self.struct['encoder_net_hidden1'], activation=self.activation,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.alpha))
            concat_layer = tf.concat([net_layer, att_layer], axis=1)
            mu = tf.layers.dense(concat_layer, self.struct['embedding_size'], activation=None,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.alpha))
            log_var = tf.layers.dense(concat_layer, self.struct['embedding_size'], activation=None,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.alpha))
        return mu, log_var

    def att_decoder_bernoulli(self, z):
        with tf.variable_scope('att_decoder', reuse=tf.AUTO_REUSE):
            h = tf.layers.dense(z, self.struct['decoder_att_hidder1'], activation=self.activation,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.alpha))
            logit = tf.layers.dense(h, self.att_dim, activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.alpha))
        return logit

    def net_decoder(self, z):
        return tf.matmul(z, z, transpose_a=False, transpose_b=True)

    def get_att_loss_gaussian(self, att_mu, att):
        return tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(att, att_mu)), axis=1))

    def get_att_loss_bernoulli(self, logit, att):
        return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=att, logits=logit), axis=1))

    def get_net_loss(self, logit):
        w_1 = (self.N * self.N - tf.reduce_sum(self.network)) / tf.reduce_sum(self.network)
        # w_2 = self.N * self.N / (self.N * self.N - tf.reduce_sum(self.network))
        return self.N * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.network, logits=logit, pos_weight=w_1))

    def reparameterization_trick(self, mu, log_var):
        z = mu + tf.random_normal((self.N, self.struct['embedding_size'])) * tf.exp(0.5 * log_var)
        return z

    def train(self):
        for epoch in range(5000):
            # for (k, input) in enumerate(self.get_att_training_data()):
            input = [x for x in range(self.N)]
            feed_dict = {self.input: input}
            net_loss, att_loss, kl_loss, reg_loss, _ = self.sess.run([self.net_loss, self.att_loss, self.kl_loss, self.reg_loss, self.train_op], feed_dict=feed_dict)
            if epoch % 20 == 0:
                #print("epoch %d, the net_loss is %f, %f， %f, %f" % (epoch, net_loss, att_loss, kl_loss, reg_loss))
                embedding = self.get_embeddings()
                # print(embedding[0])
                eval_classification(embedding, self.label)
                if self.test_lp_data is not None:
                    eval_link_prediction(embedding, self.test_lp_data)
                # if epoch > 200:
                #     self.saver.save(self.sess, 'model/embedding', global_step=epoch)
                #     np.save('model/embedding'+str(epoch)+'.npy', embedding)

    def get_embeddings(self):
        nodes = [x for x in range(self.N)]
        feed_dict = {self.input: nodes}
        embedding = self.sess.run(self.encoder_mu, feed_dict=feed_dict)
        return embedding


def train(is_lp=False):
    feature_file = prefix + '.feature'
    label_file = prefix + '.label'
    edge_file = prefix + '.edgelist'
    test_lp_data = None
    if is_lp:
        edge_file = prefix + '_lp.edgelist'
        test_lp_data = load_data.read_test_lp_data(prefix + '_lp_test')
    print(edge_file)
    feature, A, label, adj, edge_dict, N, att_dim = get_all_data(feature_file, label_file, edge_file)
    print(feature.shape)
    struct = {}
    # struct['lays_num'] = 3
    struct['encoder_net_hidden1'] = 512
    struct['encoder_att_hidden1'] = 512

    struct['embedding_size'] = 128
    struct['decoder_att_hidder1'] = 512
    vae = VAE(N, att_dim, 128, struct, A, feature, adj, label, test_lp_data)
    vae.train()
    embedding = vae.get_embeddings()
    return embedding


if __name__ == '__main__':
    train(is_lp=False)

# citeseer alpha=1.0?0.8(520), sigmoid, 72+ 128 stable ; link prediction alpha = 0.8
# cora classification: alpha=0.005?0.008(380) sigmoid 128 stable 0.85 ; link prediction alpha = 0.8
# BlogCatalog max_iter = 300(classification), max_iter link_prediction 0.784 max_iter = 1500
