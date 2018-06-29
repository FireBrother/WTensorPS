import IPython
import numpy as np

TEST_PS = False

if TEST_PS:
    import simpleflowps as sf
else:
    import simpleflow as sf

input_x = np.stack([np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)])
input_y = np.sum(input_x, axis=0)*3 + np.random.randn(input_x.shape[1])*0.5

if TEST_PS:
    DEFAULT_PS.open()

# Placeholders for training data
x = sf.Placeholder()
y_ = sf.Placeholder()

# Weigths
w = sf.Variable(np.array([[1.0, 1.0]]), name='weight')

# Threshold
b = sf.Variable(np.array([[0.0]]), name='threshold')

# Predicted class by model
y = sf.matmul(w, x) + b

loss = sf.reduce_sum(sf.square(y - y_))

train_op = sf.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

# feed_dict = {x: np.reshape(input_x, (-1, 1)), y_: np.reshape(input_y, (-1, 1))}
feed_dict = {x: input_x, y_: input_y}
with sf.Session() as sess:
    for step in range(20):
        loss_value = sess.run(loss, feed_dict=feed_dict)
        mse = loss_value / len(input_x)

        if step % 1 == 0:
            print('step: {}, loss: {}, mse: {}'.format(step, loss_value, mse))
        sess.run(train_op, feed_dict)
    w_value = sess.run(w, feed_dict=feed_dict)
    b_value = sess.run(b, feed_dict=feed_dict)
    print('w: {}, b: {}'.format(w_value, b_value))

if TEST_PS:
    DEFAULT_PS.close()