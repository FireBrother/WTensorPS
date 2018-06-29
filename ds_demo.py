#!/usr/bin/env python
# -*- coding: utf-8 -*-
import IPython
import numpy as np
import simpleflow as sf

feature = np.loadtxt('feature.txt')  # (19717, 500)
label = np.loadtxt('label.txt')  # (19717, 3)

batch = 200

X_input = sf.placeholder()
y = sf.placeholder()

npw1 = np.random.normal(0, 1, [500, 256])
npb1 = np.random.normal(0, 1, [1, 256])
npw2 = np.random.normal(0, 1, [256, 3])
npb2 = np.random.normal(0, 1, [1, 3])

w1 = sf.Variable(npw1, name='weight')
b1 = sf.Variable(npb1, name='bias')
h_fc1 = sf.relu(sf.matmul(X_input, w1) + b1)

w2 = sf.Variable(npw2, name='weight2')
b2 = sf.Variable(npb2, name='bias2')
logits = sf.softmax(sf.matmul(h_fc1, w2) + b2)

loss = sf.softmax_cross_entropy(logits, y)

train = sf.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

feed_dict = {X_input: feature, y: label}

with sf.Session() as sess:
    for step in range(10):
        loss_value = sess.run(loss, feed_dict=feed_dict)
        y_ = sess.run(logits, feed_dict=feed_dict)
        max_possibility = np.argmax(y_)
        correct_pred = np.equal(max_possibility, np.argmax(y))
        accuracy = np.mean(correct_pred)
        print('step: {}, loss: {}, acc: {}'.format(step, loss_value, accuracy))
        sess.run(train, feed_dict)
    # Create a session to run the graph
