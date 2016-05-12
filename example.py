
import numpy as np
import tensorflow as tf

from slim import ops
from slim import scopes
from slim import variables
from net_reform import NetReform

def build_a_new_graph():
    with tf.Graph().as_default() as graph:
        x = tf.placeholder(tf.float32, shape=[None, 784])
        tf.add_to_collection('input', x)
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        tf.add_to_collection('label', y_)
        x_image = tf.reshape(x, [-1,28,28,1])
        net = ops.conv2d(x_image, 64, [5, 5], scope='conv1')
        net = ops.max_pool(net, [2, 2], scope='pool1')
        net = ops.conv2d(net, 64, [5, 5], scope='conv2')
        net = ops.conv2d(net, 64, [5, 5], scope='conv2_new', stddev=0.1, bias=0.1)
        net = ops.max_pool(net, [2, 2], scope='pool2')
        net = ops.flatten(net, scope='pool2_flat')
        net = ops.fc(net, 1024, scope='fc1')
        net = ops.fc(net, 1024, scope='fc1_new')
        net = ops.fc(net, 10, activation=None, scope='fc2')
        y_conv = tf.nn.softmax(net)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        model = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        tf.add_to_collection('objective', model)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.add_to_collection('accuracy', accuracy)
        return graph

if __name__ == '__main__':
    """ NetReform Class Test """
    # Previous model 
    model = './my-model-500.meta'
    weight = './my-model-500'
    # New model to train
    new_graph = build_a_new_graph()
    # Network reform
    nr = NetReform(model, weight, new_graph)
    graph, sess = nr.reform()

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    batch = mnist.train.next_batch(100)

    obj_fn = graph.get_collection('objective')[0]
    inputs = graph.get_collection('input')[0]
    labels = graph.get_collection('label')[0]
    acc = graph.get_collection('accuracy')[0]
    
    for i in range(1000):
        batch = mnist.train.next_batch(100)
        sess.run( obj_fn, feed_dict={inputs: batch[0], labels:batch[1]} )
        if i % 100 == 0 :
            accu = sess.run( acc, feed_dict={inputs: mnist.test.images, labels: mnist.test.labels} )
            print '[Iter: {}] Validation Accuracy : {:.4f}'.format(i,accu)
    

