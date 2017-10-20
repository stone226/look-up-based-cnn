import sys
import os
import argparse
import cv2
import time
import yaml
import numpy as np
import logging
import tensorflow as tf
from tensorflow.python.client import timeline

from layers.LookupConvolution2d import extract_dense_weights
from utils import get_dataset_sizes
from networks.alexnet import alexnet_model


config = tf.ConfigProto(
    allow_soft_placement=False,
    log_device_placement=False,
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')


# test
# data:mnist(24*24,resize from 28*28)
# net:alexnet-lcnn
# delete some codes
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tensorflow Inference using LCNN.')
    parser.add_argument('--imgpath', type=str, default='/home/tm/dir/tf-lcnn-master/mnistToimage/test10k/7.jpg')
    parser.add_argument('--benchmark', type=int, default=10)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--conf', default='/home/tm/dir/tf-lcnn-master/confs/alexnet.yaml', help='configuration file path')
    parser.add_argument('--dataset', default='mnist', help='mnist, mnist224, ilsvrc2012')
    parser.add_argument('--model-conf', default='lcnnfast', help='lcnnbest, lcnn0.9, normal')  # lcnntest
    parser.add_argument('--conv', default='lcnn', help='lcnn, conv')
    args = parser.parse_args()

    # load config
    '''
    with open(os.path.join(args.path, 'conf.json'), 'r') as stream:
        conf = yaml.load(stream)
    '''
    #change
    with open(args.conf, 'r') as stream:
        conf = yaml.load(stream)

    #class_size, _ = get_dataset_sizes(conf['dataset'])
    class_size, dataset_size = get_dataset_sizes(args.dataset)
    #model_conf = {key: conf.get(key, []) for key in ['initial_sparsity', 'dictionary', 'lambda']}
    model_conf = conf['model_conf'][args.model_conf]

    #for reszie 24
    image_w = image_h = 24
    print('-----------------------------------------hhhhhhhhhhhhhhhhhhhhhh')
    # load img
    logging.info('load image')
    img = cv2.imread(args.imgpath)
    img = cv2.resize(img,(24,24), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('t',img)
    cv2.waitKey(10)
    img = img.reshape((-1, image_w, image_h, 3))


    # prepare dense network
    logging.info('prepare network')
    g1 = tf.Graph()
    with g1.as_default() as g:
        with tf.device('/cpu:0'):

            x = tf.placeholder(tf.float32, shape=[1, 24, 24, 3])
            y = tf.placeholder(tf.int64, shape=[1])
            # create network graph
            model = alexnet_model(x, class_size=class_size, convtype=args.conv, model_conf=model_conf, keep_prob=1.0)
            softmax = tf.nn.softmax(model)

    print('---------------------------------------------------------------------------------g1')
    modelpath = '/home/tm/dir/model_Lcnn_mnist/model_mnist_lcnnfast'
    with tf.Session(config=config, graph=g1) as sess:
        logging.info('start to restore - dense')
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        logging.info('start inference - dense')
        # warmup
        output = sess.run(softmax, feed_dict={x: img})
        print('predicted class = %d' % (np.argmax(output)))

        if args.conv == 'lcnn':
            gr = tf.get_default_graph()
            tensors = [gr.get_tensor_by_name('layer%d/align_conv/kernel:0' % (convid + 1)) for convid in range(7)]
            kernel_vals = sess.run(tensors)
            #logging.info('lcnn-densities: ' + ', '.join(['%.3f' % (np.count_nonzero(kernel_val) / kernel_val.size) for kernel_val in kernel_vals]))
            print('lcnn-densities:')
            for kernel_val in kernel_vals:
                print(np.count_nonzero(kernel_val) / kernel_val.size)
            '''
        elapsed = 0
        for _ in range(args.benchmark):
            t = time.time()
            output = sess.run([model], feed_dict={
                x: img
            })
            elapsed += time.time() - t
        logging.info('average elapsed time(dense) = %f' % (elapsed / args.benchmark))
            '''
        extract_dense_weights(sess)

    tf.reset_default_graph()

    if args.conv == 'conv':
        sys.exit(0)
    print('-------------------------------------------------------------------------------------------g2')
    g2 = tf.Graph()
    with g2.as_default() as g:
        with tf.device('/cpu:0'):

            x = tf.placeholder(tf.float32, shape=[1, 24, 24, 3])
            y = tf.placeholder(tf.int64, shape=[1])
            # create network graph
            model = alexnet_model(x, class_size=class_size, convtype=args.conv, model_conf=model_conf,keep_prob=1.0)
            softmax = tf.nn.softmax(model)

    print('-------------------sssssssssssssssss')
    with tf.Session(config=config, graph=g2) as sess:
        logging.info('start to restore - sparse')
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        logging.info('start inference - sparse')

        # warmup
        output = sess.run([softmax], feed_dict={x: img})
        print('predicted class = %d' %(np.argmax(output)))
        '''
        elapsed = 0
        for _ in range(args.benchmark):
            t = time.time()
            output = sess.run([model], feed_dict={x: img
            })
            elapsed += time.time() - t
        logging.info('average elapsed time(sparse) = %f' % (elapsed / args.benchmark))
        '''