from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils


HParams = namedtuple('HParams',
                    'batch_size, num_classes, num_residual_units, k, '
                    'weight_decay, initial_lr, decay_step, lr_decay, '
                    'momentum, no_logit_map')


class ResNet(object):
    def __init__(self, hp, images, labels, global_step):
        self._hp = hp # Hyperparameters
        self._images = images # Input image
        self._labels = labels
        self._global_step = global_step
        self.is_train = tf.placeholder(tf.bool)

    def set_clustering(self, clustering):
        print('Parse clustering')
        cluster_size = [[len(sublist2) for sublist2 in sublist1] for sublist1 in clustering]
        self._split2 = [item for sublist in cluster_size for item in sublist]
        self._split1 = [sum(sublist) for sublist in cluster_size]
        self._logit_map = [item for sublist1 in clustering for sublist2 in sublist1 for item in sublist2]
        print('\t1st level: %d splits %s' % (len(self._split1), self._split1))
        print('\t2nd level: %d splits %s' % (len(self._split2), self._split2))
        # print self._logit_map

    def _split_channels(self, N, groups):
        group_total = sum(groups)
        float_outputs = [float(N)*t/group_total for t in groups]
        for i in xrange(1, len(float_outputs), 1):
            float_outputs[i] = float_outputs[i-1] + float_outputs[i]
        outputs = map(int, map(round, float_outputs))
        for i in xrange(len(outputs)-1, 0, -1):
            outputs[i] = outputs[i] - outputs[i-1]
        return outputs

    def build_model(self):
        print('Building model')
        # Init. conv.
        print('\tBuilding unit: init_conv')
        x = utils._conv(self._images, 3, 16, 1, name='init_conv')

        # Residual Blocks
        filters = [16 * self._hp.k, 32 * self._hp.k, 64 * self._hp.k]
        strides = [1, 2, 2]

        filter1_split1 = self._split_channels(filters[1], self._split1)
        filter2_split1 = self._split_channels(filters[2], self._split1)
        filter2_split2 = self._split_channels(filters[2], self._split2)

        x = self.residual_block_first(x, filters[0], strides[0], 'unit_1_0')
        for j in xrange(1, self._hp.num_residual_units, 1):
            x = self.residual_block(x, 'unit_1_%d' % (j))
        x = self.residual_block_first(x, filters[1], strides[1], 'unit_2_0')
        for j in xrange(1, self._hp.num_residual_units, 1):
            x = self.residual_block(x, 'unit_2_%d' % (j))
        x = self.residual_block_first_split(x, filter1_split1, filter2_split1, strides[2], 'unit_3_0')
        for j in xrange(1, self._hp.num_residual_units, 1):
            x = self.residual_block_split(x, filter2_split1, 'unit_3_%d' % (j))

        # Last unit
        with tf.variable_scope('unit_last') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = utils._bn(x, self.is_train, self._global_step)
            x = utils._relu(x)
            x = tf.reduce_mean(x, [1, 2])

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, x_shape[1]])
            x = self.fc_split(x, filter2_split2, self._split2)
            if not self._hp.no_logit_map:
                x = tf.transpose(tf.gather(tf.transpose(x), self._logit_map))

        self._logits = x

        # Probs & preds & acc
        self.probs = tf.nn.softmax(x, name='probs')
        self.preds = tf.to_int32(tf.argmax(self._logits, 1, name='preds'))
        ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
        zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
        correct = tf.select(tf.equal(self.preds, self._labels), ones, zeros)
        self.acc = tf.reduce_mean(correct, name='acc')
        tf.scalar_summary('accuracy', self.acc)

        # Loss & acc
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(x, self._labels)
        self.loss = tf.reduce_mean(loss, name='cross_entropy')
        tf.scalar_summary('cross_entropy', self.loss)


    def residual_block_first(self, x, out_channel, strides, name='unit'):
        # First residual unit
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            x = utils._bn(x, self.is_train, self._global_step, name='bn_1')
            x = utils._relu(x, name='relu_1')

            in_channel = x.get_shape().as_list()[-1]
            # Shortcut
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1],
                                              [1, strides, strides, 1], 'VALID')
            else:
                shortcut = utils._conv(x, strides, out_channel, strides, name='shortcut')

            # Residual
            x = utils._conv(x, 3, out_channel, strides, name='conv_1')
            x = utils._bn(x, self.is_train, self._global_step, name='bn_2')
            x = utils._relu(x, name='relu_2')
            x = utils._conv(x, 3, out_channel, 1, name='conv_2')

            # Merge
            x = x + shortcut
        return x


    def residual_block_first_split(self, x, in_splits, out_splits, strides, name='unit'):
        b, h, w, num_channel = x.get_shape().as_list()
        assert num_channel == sum(in_splits)
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s with %d splits' % (scope.name, len(in_splits)))
            outs = []
            offset_in = 0
            for i, (n_in, n_out) in enumerate(zip(in_splits, out_splits)):
                sliced = tf.slice(x, [0, 0, 0, offset_in], [b, h, w, n_in])
                sliced_residual = self.residual_block_first(sliced, n_out, strides, name=('split_%d' % (i+1)))
                outs.append(sliced_residual)
                offset_in += n_in
            concat = tf.concat(3, outs)
        return concat


    def residual_block(self, x, name='unit'):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut
            shortcut = x

            # Residual
            x = utils._bn(x, self.is_train, self._global_step, name='bn_1')
            x = utils._relu(x, name='relu_1')
            x = utils._conv(x, 3, num_channel, 1, name='conv_1')
            x = utils._bn(x, self.is_train, self._global_step, name='bn_2')
            x = utils._relu(x, name='relu_2')
            x = utils._conv(x, 3, num_channel, 1, name='conv_2')

            # Merge
            x = x + shortcut
        return x


    def residual_block_split(self, x, splits, name='unit'):
        b, h, w, num_channel = x.get_shape().as_list()
        assert num_channel == sum(splits)
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s with %d splits' % (scope.name, len(splits)))
            outs = []
            offset = 0
            for i, n in enumerate(splits):
                sliced = tf.slice(x, [0, 0, 0, offset], [b, h, w, n])
                sliced_residual = self.residual_block(sliced, name=('split_%d' % (i+1)))
                outs.append(sliced_residual)
                offset += n
            concat = tf.concat(3, outs)
        return concat


    def fc_split(self, x, in_splits, out_splits, name='unit'):
        b, num_in = x.get_shape().as_list()
        assert num_in == sum(in_splits)
        with tf.variable_scope(name) as scope:
            print('\tBuilding fc layer: %s with %d splits' % (scope.name, len(in_splits)))
            outs = []
            offset_in = 0
            for i, (n_in, n_out) in enumerate(zip(in_splits, out_splits)):
                sliced = tf.slice(x, [0, offset_in], [b, n_in])
                sliced_fc = utils._fc(sliced, n_out, name="split_%d" % (i+1))
                outs.append(sliced_fc)
                offset_in += n_in
            concat = tf.concat(1, outs)
        return concat


    def build_train_op(self):
        # Add l2 loss
        with tf.variable_scope('l2_loss'):
            costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
            # for var in tf.get_collection(utils.WEIGHT_DECAY_KEY):
                # tf.histogram_summary(var.op.name, var)
            l2_loss = tf.mul(self._hp.weight_decay, tf.add_n(costs))
        self._total_loss = self.loss + l2_loss

        # Learning rate
        self.lr = tf.train.exponential_decay(self._hp.initial_lr, self._global_step,
                                        self._hp.decay_step, self._hp.lr_decay, staircase=True)
        tf.scalar_summary('learing_rate', self.lr)

        # Gradient descent step
        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        grads_and_vars = opt.compute_gradients(self._total_loss, tf.trainable_variables())
        # print '\n'.join([t.name for t in tf.trainable_variables()])
        apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

        # Batch normalization moving average update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            with tf.control_dependencies(update_ops+[apply_grad_op]):
                self.train_op = tf.no_op()
        else:
            self.train_op = apply_grad_op
