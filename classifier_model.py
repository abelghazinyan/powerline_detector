import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_manager import DataManagerConvolution

data_manager = DataManagerConvolution()
data_manager.load_from_file()
data_manager.set_up()


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_2by2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name)


def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


x = tf.placeholder(tf.float32, shape=[None, 40, 40, 3])
y_true = tf.placeholder(tf.float32, shape=[None, 2])

hold_prob = tf.placeholder(tf.float32)

# Decoder
convo_1 = convolutional_layer(x, shape=[3, 3, 3, 6])
convo_1_pooling = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, shape=[3, 3, 6, 12])
convo_2_pooling = max_pool_2by2(convo_2)

convo_3 = convolutional_layer(convo_2_pooling, shape=[3, 3, 12, 24])
convo_3_pooling = max_pool_2by2(convo_3)

# decoder
convo_3_flat = tf.reshape(convo_3_pooling, [-1, 5 * 5 * 24])

full_layer_one = tf.nn.relu(normal_full_layer(convo_3_flat, 1024))

full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout, 2)

# Loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1):
        batch = data_manager.next_batch(10)

        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})

        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i % 1 == 0:
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(acc, feed_dict={x: data_manager.X_test, y_true: data_manager.y_test, hold_prob: 0.5}))
            print('\n')
    saver.save(sess, "/tmp/classifier_model.ckpt")
