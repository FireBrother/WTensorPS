#!/usr/bin/env python
# -*- coding: utf-8 -*-
import IPython
import numpy as np
import simpleflow as sf

feature = np.loadtxt('feature.txt').T  # (19717, 500)
label = np.loadtxt('label.txt').T  # (19717, 3)

batch = 200

X_input = sf.placeholder()
y = sf.placeholder()

npw1 = np.random.normal(0, 1, [256, 500])
npb1 = np.random.normal(0, 1, [256, 1])
npw2 = np.random.normal(0, 1, [3, 256])
npb2 = np.random.normal(0, 1, [3, 1])

w1 = sf.Variable(npw1, name='weight')
b1 = sf.Variable(npb1, name='bias')
h_fc1 = sf.relu(sf.matmul(w1, X_input) + b1)

w2 = sf.Variable(npw2, name='weight2')
b2 = sf.Variable(npb2, name='bias2')
h_fc2 = sf.matmul(w2, h_fc1) + b2

loss = sf.softmax_cross_entropy(h_fc2, y)

train = sf.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

feed_dict = {X_input: feature, y: label}

with sf.Session() as sess:
    # IPython.embed(), quit()
    for step in range(10):
        loss_value = sess.run(loss, feed_dict=feed_dict)
        loss_value = np.mean(loss_value)
        y_ = sess.run(h_fc2, feed_dict=feed_dict)
        max_possibility = np.argmax(y_, axis=0)
        correct_pred = np.equal(max_possibility, np.argmax(label, axis=0))
        accuracy = np.mean(correct_pred)
        print('step: {}, loss: {}, acc: {}'.format(step, loss_value, accuracy))
        sess.run(train, feed_dict)
    # Create a session to run the graph
