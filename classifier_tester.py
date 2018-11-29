import tensorflow as tf
import os, os.path
import io
import numpy as np
import matplotlib.pyplot as plt
from data_manager import DataManagerConvolution

data_manager = DataManagerConvolution()
data_manager.load_from_file()
data_manager.set_up()

index = 29
img = data_manager.get_img(index)

if data_manager.get_label(index)[0] == 1.:
    print("Wire")
else:
    print("Not Wire")
#
# plt.imshow(img)
# plt.axis("off")
# plt.show()

img = np.expand_dims(img, axis=0)

tf.reset_default_graph()
with tf.Session() as sess:

    new_saver = tf.train.import_meta_graph('model_classifier/classifier_model.meta')

    new_saver.restore(sess, tf.train.latest_checkpoint('model_classifier/'))

    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name("x:0")
    prediction = graph.get_tensor_by_name("prediction:0")
    hold_prob = graph.get_tensor_by_name("hold_prob:0")
    # normalizing input
    img = img / 255

    pred = sess.run(tf.nn.softmax(prediction), feed_dict={input: img, hold_prob: 0.5})

    if pred[0][0] > pred[0][1]:
        print("Wire {}%".format(pred[0][0] * 100))
    else:
        print("Not Wire {}%".format(pred[0][1] * 100))
