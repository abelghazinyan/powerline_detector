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


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2by2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME', name=name)


def convolutional_layer(input_x, shape, name):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.identity(tf.nn.relu(conv2d(input_x, W) + b), name=name)


def normal_full_layer(input_layer, size, name):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.add(tf.matmul(input_layer, W), b, name=name)


x = tf.placeholder(tf.float32, shape=[None, 40, 40, 3], name="x")
y_true = tf.placeholder(tf.float32, shape=[None, 2], name="y_true")

hold_prob = tf.placeholder(tf.float32, name="hold_prob")

# Decoder
convo_1 = convolutional_layer(x, [3, 3, 3, 6], "convo_1")
convo_1_pooling = max_pool_2by2(convo_1, "max_pool_1")

convo_2 = convolutional_layer(convo_1_pooling, [3, 3, 6, 12], "convo_2")
convo_2_pooling = max_pool_2by2(convo_2, "max_pool_2")

convo_3 = convolutional_layer(convo_2_pooling, [3, 3, 12, 24], "convo_3")
convo_3_pooling = max_pool_2by2(convo_3, "max_pool_3")

# decoder
convo_3_flat = tf.identity(tf.reshape(convo_3_pooling, [-1, 5 * 5 * 24]), name="convo_3_flat")

full_layer_one = tf.nn.relu(normal_full_layer(convo_3_flat, 1024, "full_1"))

full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout, 2, "prediction")

# Loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = data_manager.next_batch(128)

        _, loss = sess.run([train, cross_entropy], feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})

        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i % 100 == 0:
            print('Currently on step {}'.format(i))
            print('Loss is:')
            print(loss)
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            print(sess.run(acc, feed_dict={x: data_manager.X_test, y_true: data_manager.y_test, hold_prob: 0.5}))
            print('\n')
    save_path = saver.save(sess, "/model_classifier/classifier_model",
                           write_meta_graph=True)