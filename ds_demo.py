#!/usr/bin/env python
# -*- coding: utf-8 -*-
import IPython
import numpy as np
import sys

import simpleflowps as sf

feature = np.loadtxt(sys.argv[1])  # (19717, 500)
label = np.loadtxt(sys.argv[2])  # (19717, 3)

DEFAULT_PS.open()

X_input = sf.placeholder()
y = sf.placeholder()

npw1 = np.random.normal(0, 0.1, [500, 256])
npb1 = np.ones([1, 256]) * 0.1
npw2 = np.random.normal(0, 0.1, [256, 3])
npb2 = np.ones([1, 3]) * 0.1

w1 = sf.Variable(npw1, name='weight')
b1 = sf.Variable(npb1, name='bias')
h_fc1 = sf.relu(sf.matmul(X_input, w1) + b1)

w2 = sf.Variable(npw2, name='weight2')
b2 = sf.Variable(npb2, name='bias2')
h_fc2 = sf.matmul(h_fc1, w2) + b2

loss = sf.softmax_cross_entropy(h_fc2, y)

train = sf.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

feed_dict = {X_input: feature, y: label}

with sf.Session() as sess:
    # IPython.embed(), quit()
    for step in range(1000):
        index = np.arange(feature.shape[0])
        np.random.shuffle(index)
        feature = feature[index]
        label = label[index]
        loss_value, y_, _ = sess.multi_run([loss, h_fc2, train], feed_dict=feed_dict)
        # loss_value = sess.run(loss, feed_dict=feed_dict)
        # y_ = sess.run(h_fc2, feed_dict=feed_dict)
        # sess.run(train, feed_dict=feed_dict)
        loss_value = np.mean(loss_value)
        max_possibility = np.argmax(y_, axis=1)
        correct_pred = np.equal(max_possibility, np.argmax(label, axis=1))
        accuracy = np.mean(correct_pred)
        # if step % 100 == 0:
        print('step: {}, loss: {}, acc: {}'.format(step, loss_value, accuracy))

        # params = sess.multi_run([w1, b1], feed_dict)
        # print(params)
        # input()
    # Create a session to run the graph

DEFAULT_PS.close()
