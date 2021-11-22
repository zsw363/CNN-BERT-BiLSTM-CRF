# -*- coding utf-8 -*-
"""
Create on 2020/12/19 14:49
@author: zsw
"""
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

class BLSTM_CRF(object):
    def __init__(self, embedded_chars, hidden_unit, cell_type, num_layers, dropout_rate,
                 initializers, num_labels, seq_length, labels, lengths, is_training,
                 cnn_input = 0, is_cnn = False, ):
        """
        BLSTM-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param hidden_unit: LSTM的隐含单元个数
        :param cell_type: RNN类型（LSTM OR GRU DICNN will be add in feature）
        :param num_layers: RNN的层数
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        self.hidden_unit = hidden_unit
        self.dropout_rate = dropout_rate
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training
        self.is_cnn = is_cnn
        self.input_picture = cnn_input
        self.image_height = 100
        self.image_width = 100

    def create_cnn(self, X, b_alpha=0.1, keep_prob=0.75, image_height=100, image_width=100, emb_size=256,name='cnn_'):
        #input: X:[v]*batch_size*height*width*seq_length
        #return:[batch_size, lengths*emb_size]

        # a = tf.reshape(X, shape=[-1, self.seq_length, image_height, image_width])
        # batch_size = a.shape[0]
        # a = a.eval()
        # for b in a:
        #     print(b.shape)
        #     exit()
        with tf.name_scope(name) as scope:
            #create_model
            x = tf.reshape(X, shape=[-1, image_height, image_width, 1])
            x = tf.to_float(x)
            # print(">>> input x: {}".format(x))


            # 卷积层1
            wc1 = tf.get_variable(name=scope+'wc1',shape=[3, 3, 1, 32], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            bc1 = tf.Variable(b_alpha * tf.random_normal([32]))
            conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 3, 3, 1], padding='SAME'), bc1))
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv1 = tf.nn.dropout(conv1, keep_prob)

            # 卷积层2
            wc2 = tf.get_variable(name=scope+'wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            bc2 = tf.Variable(b_alpha * tf.random_normal([64]))
            conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 3, 3, 1], padding='SAME'), bc2))
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv2 = tf.nn.dropout(conv2, keep_prob)

            # 卷积层3
            wc3 = tf.get_variable(name=scope+'wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            bc3 = tf.Variable(b_alpha * tf.random_normal([128]))
            conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 3, 3, 1], padding='SAME'), bc3))
            conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv3 = tf.nn.dropout(conv3, keep_prob)
            # print(">>> convolution 3: ", conv3.shape)
            next_shape = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]

            # 全连接层1
            wd1 = tf.get_variable(name=scope+'wd1', shape=[next_shape, 1024], dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
            bd1 = tf.Variable(b_alpha * tf.random_normal([1024]))
            dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
            dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
            dense = tf.nn.dropout(dense, keep_prob)
            # 全连接层2
            wout = tf.get_variable(scope+'name', shape=[1024, emb_size], dtype=tf.float32,
                                       initializer=tf.contrib.layers.xavier_initializer())
            bout = tf.Variable(b_alpha * tf.random_normal([emb_size]))

            with tf.name_scope(scope+'y_prediction'):
                y_predict = tf.add(tf.matmul(dense, wout), bout)

        return y_predict


    def add_blstm_crf_layer(self, crf_only):
        """
        blstm-crf网络
        :return:
        """
        if self.is_training:
            # lstm input dropout rate i set 0.9 will get best score
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.dropout_rate)

        if crf_only:
            logits = self.project_crf_layer(self.embedded_chars)
        else:
            # blstm
            lstm_output = self.blstm_layer(self.embedded_chars)
            # project
            logits = self.project_bilstm_layer(lstm_output)

            self.labels = tf.reshape(self.labels, [-1,1])
            newlogits = tf.reshape(logits, [-1, self.seq_length*self.num_labels])
            with tf.variable_scope("output"):
                W = tf.get_variable("W",shape=[self.seq_length*self.num_labels, self.num_labels],
                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                self.scores = tf.nn.xw_plus_b(newlogits, W, b)
                logits = tf.reshape(self.scores, shape=[-1, 1, self.num_labels])

            # crf
            loss, trans, per_example_loss = self.crf_layer(logits)
            # CRF decode, pred_ids 是一条最大概率的标注路径
            pred_ids, scores = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=1)

            return (loss, logits, per_example_loss, pred_ids, self.weight)
        # if not self.use_crf:
        #     #全连接
        #     newlogits = tf.reshape(logits, [-1, self.seq_length*self.num_labels])
        #     with tf.name_scope("output"):
        #         W = tf.get_variable(
        #             "W",
        #             shape=[self.seq_length*self.num_labels, self.num_labels],
        #             initializer=tf.contrib.layers.xavier_initializer())
        #         b = tf.Variable(tf.constant(0.1, shape=[self.num_labels]), name="b")
        #         self.scores = tf.nn.xw_plus_b(newlogits, W, b, name="scores")
        #         self.predictions = tf.argmax(self.scores, -1, name="predictions")
        #     # CalculateMean cross-entropy loss
        #     with tf.name_scope("loss"):
        #         per_example_loss = -tf.nn.softmax_cross_entropy_with_logits(logits=tf.cast(self.predictions,dtype=tf.float32), labels=self.labels)
        #         loss = tf.reduce_mean(per_example_loss)
        #     return (loss, self.scores, per_example_loss, self.predictions)
        # else:
        #     # crf
        #     loss, trans, per_example_loss = self.crf_layer(logits)
        #     # CRF decode, pred_ids 是一条最大概率的标注路径
        #     pred_ids, scores = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        #     return (loss, logits, per_example_loss, pred_ids)

    def _witch_cell(self):
        """
        RNN 类型
        :return:
        """
        cell_tmp = None
        if self.cell_type == 'lstm':
            cell_tmp = rnn.LSTMCell(self.hidden_unit)
        elif self.cell_type == 'gru':
            cell_tmp = rnn.GRUCell(self.hidden_unit)
        return cell_tmp

    def _bi_dir_rnn(self):
        """
        双向RNN
        :return:
        """
        cell_fw = self._witch_cell()
        cell_bw = self._witch_cell()
        if self.dropout_rate is not None:
            cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_rate)
            cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_rate)
        return cell_fw, cell_bw

    def blstm_layer(self, embedding_chars):
        """

        :return:
        """
        with tf.variable_scope('rnn_layer'):
            cell_fw, cell_bw = self._bi_dir_rnn()
            if self.num_layers > 1:
                cell_fw = rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
                cell_bw = rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embedding_chars,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)
        return outputs

    def project_bilstm_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        if self.is_cnn:
            cnn_inputs = tf.reshape(self.input_picture, shape=[-1, self.seq_length, self.image_height, self.image_width])
            cnn_inputs = tf.split(cnn_inputs, self.seq_length, 1)
            # print(cnn_inputs)
            y_predicts = []
            for (i, token_input) in enumerate(cnn_inputs):
                    pre = self.create_cnn(token_input, emb_size=256, name='cnn_'+str(i))
                    pre = tf.reshape(pre, [-1, 1, 256])
                    y_predicts.append(pre)
            cnn_output = tf.concat(y_predicts, 1)
            # print(cnn_output.shape)
            # exit()
            # y_predict = self.create_cnn(self.input_picture, emb_size=256)
            # cnn_output = tf.reshape(y_predict, [-1, self.seq_length, 256])
            if 0:
                cnn_output = tf.reshape(cnn_output, [-1, self.seq_length*256])
                with tf.variable_scope("forgetgate"):
                    W = tf.get_variable("W", shape=[self.seq_length*256, self.seq_length*256],
                                        dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                    b = tf.get_variable("b", shape=[self.seq_length*256], dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

                    cnn_output = tf.reshape(tf.nn.xw_plus_b(cnn_output, W, b),[-1, self.seq_length, 256])

                outputs = tf.add(lstm_outputs, cnn_output)
            else:
                outputs = tf.add(lstm_outputs, cnn_output)
        else:
            outputs = lstm_outputs

        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.hidden_unit * 2, self.hidden_unit],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.hidden_unit], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(outputs, shape=[-1, self.hidden_unit * 2])
                hidden = tf.nn.xw_plus_b(output, W, b)
            self.weight = b

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.hidden_unit, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def project_crf_layer(self, embedding_chars, name=None):
        """
        hidden layer between input layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.embedding_dims, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b", shape=[self.num_labels], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(self.embedded_chars,
                                    shape=[-1, self.embedding_dims])  # [batch_size, embedding_dims]
                pred = tf.tanh(tf.nn.xw_plus_b(output, W, b))
            return tf.reshape(pred, [-1, self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            if self.labels is None:
                return None, trans, None
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=self.labels,
                    transition_params=trans,
                    sequence_lengths=self.lengths)
                return tf.reduce_mean(-log_likelihood), trans, -log_likelihood


