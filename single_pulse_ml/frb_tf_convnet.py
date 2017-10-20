#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for FRB detection, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from sklearn.model_selection import train_test_split

import reader as reader
import plot_tools

tf.logging.set_verbosity(tf.logging.INFO)

# Check if python3 is being used
py3 = sys.version_info[0] > 2

def get_predictions(data, classifier):
  """ Take test data and a classifier and 
  return the model's predicted label and the associated 
  probability.

  e.g.:

  eval_results, clf = train_eval_1d_cnn(train_data, train_labels, eval_data, eval_labels)

  pred_labels, prob = get_predictions(eval_data, clf)

  """
  predictions = classifier.predict(data, input_fn=None)

  label, prob = [], []

  while True:
    try:
      if py3 is True:
        pred = predictions.__next__()
      else:
        pred = predictions.next()

      label.append(pred['classes'])
      prob.append(pred['probabilities'])
    except StopIteration:
      return label, np.array(prob)

def cnn_model_1d(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  # input_layer = tf.reshape(features, [-1, 28, 28, 1])
  h = int(features.shape[1])
  batch_len = features.shape[0]

  input_layer = tf.reshape(features, [-1, h, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  # conv1 = tf.layers.conv1d(
  #     inputs=input_layer,
  #     filters=32,
  #     kernel_size=5,
  #     padding="same",
  #     activation=tf.nn.relu)

  conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=32,
      strides=2,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=[2], strides=2)
  print(pool1.shape)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=64,
      kernel_size=2,
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=[2], strides=2)
  print(pool2.shape)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  #  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  # data have been pooled twice; new shape is h//4, w//4, nfeatures
#  pool1_flat = tf.reshape(pool1, [-1, h//2 * 64])
  pool2_flat = tf.reshape(pool2, [-1, h//8 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=2)
  loss = None
  train_op = None

  print(learn.ModeKeys.INFER, learn.ModeKeys.TRAIN)

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def cnn_model_2d(features, labels, mode, nfilt1=32):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # input_layer = tf.reshape(features, [-1, h, w, 1])
  h, w = int(features.shape[1]), int(features.shape[2])
  input_layer = tf.reshape(features, [-1, h, w, 1])
  print(input_layer.shape)

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=nfilt1,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  print(pool1.shape)
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=nfilt2,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  print(pool2.shape)
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  #  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  # data have been pooled twice; new shape is h//4, w//4, nfeatures
  pool2_flat = tf.reshape(pool2, [-1, h//4 * w//4 * nfilt2])
  print(pool2_flat.shape)
  print(nfilt2

    )
  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dense = tf.layers.dense(inputs=dense, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=2)
  loss = None
  train_op = None

  print(learn.ModeKeys.INFER, learn.ModeKeys.TRAIN)

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def train_eval_1d_cnn(train_data, train_labels, 
                      eval_data, eval_labels, model_dir="/tmp/frb_convnet_model"):
  """ Take 
  """

  # Create the Estimator
  mnist_classifier = learn.Estimator(
      model_fn=cnn_model_1d, model_dir=model_dir)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)


  # Train the model
  mnist_classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=50,
      steps=250,
      monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
      "precision":
          learn.MetricSpec(
              metric_fn=tf.metrics.precision, prediction_key="classes"),
      "false_negatives":
          learn.MetricSpec(
              metric_fn=tf.metrics.false_negatives, prediction_key="classes"),
      "recall":
          learn.MetricSpec(
              metric_fn=tf.metrics.recall, prediction_key="classes"),
  }

  # Evaluate the model and print results
  eval_results = mnist_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)

  return eval_results, mnist_classifier

def restor_mod(mod_name):
  """ restore model with mod_name
      e.g. restor_mod('/tmp/frb_convnet_model/model.ckpt-250.meta')
      this is not usable code.
  """
  sess=tf.Session()    
  #First let's load meta graph and restore weights
  saver = tf.train.import_meta_graph(mod_name)
  saver.restore(sess,tf.train.latest_checkpoint(mod_name.split('/')[:-1]))
  pass 

def train_eval_2d_cnn(train_data, train_labels, 
                      eval_data, eval_labels, model_dir="/tmp/frb_convnet_model",
                      nfilt2=64):
  """ Take 
  """

  # Create the Estimator
  frb_classifier = learn.Estimator(
      model_fn=cnn_model_2d, model_dir=model_dir)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  frb_classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=50,
      steps=250,
      monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
      "precision":
          learn.MetricSpec(
              metric_fn=tf.metrics.precision, prediction_key="classes"),
      "false_negatives":
          learn.MetricSpec(
              metric_fn=tf.metrics.false_negatives, prediction_key="classes"),
      "recall":
          learn.MetricSpec(
              metric_fn=tf.metrics.recall, prediction_key="classes"),
  }

  # Evaluate the model and print results
  eval_results = frb_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)

  return eval_results, frb_classifier


def run_cnn_1d(fn, sum_freq=False, nfreq=16, ntime=250, plot=True, train_size=0.75):
  """ Take npy array with training / test data of a 1d timestream, 
  build a CNN model using cnn_model_fn, print metrics, 
  return predictions for each test point along with 
  their probabilities.
  """ 
  f = np.load(fn)

  d, y = f[:, :-1], f[:, -1]

  if sum_freq is True:
    d = d.reshape(-1, nfreq, ntime).mean(1)

  #  assert (f.shape[-1]-1)==nfreq*ntime, "Input nfreq, ntime are wrong"
  
  # rfi = y==0
  # d[rfi] = np.random.normal(0, 1, 8551*250).reshape(-1, 250)

  # Split data up randomly, using train_size as the fraction of 
  # total events to train off of
  train_data, eval_data, train_labels, eval_labels = \
              train_test_split(d, y, train_size=train_size)

  print("Training on %d triggers" % len(train_data))

  train_data = train_data.astype(np.float32)[..., None]
  eval_data = eval_data.astype(np.float32)[..., None]
  train_labels = train_labels.astype(np.int32)
  eval_labels = eval_labels.astype(np.int32)

  eval_results, clf = train_eval_1d_cnn(train_data, train_labels, eval_data, eval_labels)

  pred_labels, prob = get_predictions(eval_data, clf)

  indy = np.where(pred_labels!=eval_labels)[0]
  print(indy)

  if plot is True:
    plot_tools.plot_ranked_triggers(eval_data, prob, h=12, w=12)
    plot_tools.plot_ranked_triggers(eval_data, prob, h=12, w=12, ascending=True)
    plot_tools.plot_ranked_triggers(eval_data, prob, h=12, w=12, ascending='mid')

  return pred_labels, prob, eval_results, clf, eval_data, pred_labels, eval_labels

def estimate_time_complexity_tree(n_t, n_nu):

  return n_t * n_nu * np.log2(n_nu) / 1e6 

def estimate_time_complexity(n_nu=16, n_t=250, n_k1=32, n_k2=64, 
                             n_r=2, n_d1=1024, n_d2=1024, n_out=2,
                             print_complexity=False):

  o1 = (n_nu*n_t*np.log2(n_t)**2 / 1e6)
  o2 = (n_nu*n_t*n_k1 / 1e6)
  o3 = (n_nu * n_t / n_r**2 * n_k2 * np.log2(n_t / n_r) / 1e6)
  o4 = (n_nu * n_t / n_r**2 * n_k2 / 1e6)
  o5 = (n_nu * n_t / n_r**4 * n_k2 * n_d1 / 1e6)
  o6 = (n_k2**2 * n_d1 / 1e6)
  o7 = (n_d1**2 * n_d2 / 1e6)
  o8 = (n_d2**2 * n_out / 1e6)

  if print_complexity == True:
    print("n_nu * n_t * log(n_t)**2 = %f" % (n_nu*n_t*np.log2(n_t)**2 / 1e6))

    print("n_nu * n_t * n_k1 = %f" % (n_nu*n_t*n_k1 / 1e6))

    print("n_nu * n_t / n_r**2 * n_k2 * log(n_t/n_r)= %d" % \
      (n_nu * n_t / n_r**2 * n_k2 * np.log2(n_t / n_r) / 1e6))

    print("n_nu * n_t / n_r**2 * n_k2 = %f" % (n_nu * n_t / n_r**2 * n_k2 / 1e6))

    print("n_nu * n_t / n_r**4 * n_k2 * n_d1 = %f" % (n_nu * n_t / n_r**4 * n_k2 * n_d1 / 1e6))

    print("n_k2**2 * n_d1 = %f" % (n_k2**2 * n_d1 / 1e6))

    print("n_d1**2 * n_d2 = %f" % (n_d1**2 * n_d2 / 1e6))

    print("n_d2**2 * n_out = %f" % (n_d2**2 * n_out / 1e6))


  return o1,o2,o3,o4,o5,o6,o7

def save_results(fout, results):
  import pickle

  print(results)
  f = open(fout, 'wb')
#  print(json.dumps(results))
#  json.dump(json.dumps(results), f)
#  f.close()

  pickle.dump(results, f, protocol=2)
  f.close()

  print("Wrote to json file: %s" % fout)


def restore_model(model_fn=cnn_model_2d, 
            model_dir='/tmp/frb_convnet_modelwidth8/'):
  """ Check meta data in model_dir and load 
  model
  """

  clf = learn.Estimator(model_fn=model_fn, model_dir=model_dir)

  return clf


def run_cnn_2d(fn, nfreq=16, ntime=250, plot=True, 
                train_size=0.75, twidth=32, train_only=False,
                model_dir="/tmp/frb_convnet_model"):
  """ Take npy array with training / test data, 
  build a CNN model using cnn_model_fn, print metrics, 
  return predictions for each test point along with 
  their probabilities.
  """ 
  f = np.load(fn)

  assert (f.shape[-1]-1)==nfreq*ntime, "Input nfreq, ntime are wrong"
  assert twidth <= ntime//2, "twidth should be half ntime or smaller"

  # Split data up randomly, using train_size as the fraction of 
  # total events to train off of
  train_data, eval_data, train_labels, eval_labels = \
              train_test_split(f[:, :-1], f[:, -1], train_size=train_size)

#  eval_data, train_data, eval_labels, train_labels = f[1::2,:-1],f[::2, :-1],f[1::2,-1],f[::2, -1]
  
  # Try evaluating on a different population of FRBs than were trained on
  # In this case half of the FRB triggers have spec in <-3 and the other half >+3

  # eval_data = np.concatenate((f[:, :-1][:8551//2], f[:, :-1][8600:12000]), axis=0)[::2]
  # eval_labels = np.concatenate((f[:, -1][:8551//2], f[:, -1][8600:12000]))[::2]

  # train_data = np.concatenate((f[:, :-1][8551//2:8551], f[:, :-1][-4400:]), axis=0)[::2]
  # train_labels = np.concatenate((f[:, -1][8551//2:8551], f[:, -1][-4400:]))[::2]

  print("Training on %d triggers" % len(train_data))

  train_data = train_data.astype(np.float32)
  eval_data = eval_data.astype(np.float32)  
  train_labels = train_labels.astype(np.int32)
  eval_labels = eval_labels.astype(np.int32)

  train_data = train_data.reshape(-1, nfreq, ntime)[:, :, ntime//2-twidth:ntime//2+twidth]
  eval_data = eval_data.reshape(-1, nfreq, ntime)[:, :, ntime//2-twidth:ntime//2+twidth]

  eval_results, clf = train_eval_2d_cnn(train_data, train_labels, \
                              eval_data, eval_labels, model_dir=model_dir)

  if train_only is True:
    return clf

  pred_labels, prob = get_predictions(eval_data, clf)

  indy = np.where(pred_labels!=eval_labels)[0]
  print(indy)

  if plot is True:
    plot_tools.plot_ranked_triggers(eval_data, prob, h=12, w=12)
    plot_tools.plot_ranked_triggers(eval_data, prob, h=12, w=12, ascending=True)
    plot_tools.plot_ranked_triggers(eval_data, prob, h=12, w=12, ascending='mid')

  return pred_labels, prob, eval_results, clf, eval_data, pred_labels, eval_labels

def classify_dataset(fn, model_dir, nfreq=16, ntime=250, twidth=32):

  # Load the trained model from model_dir
  clf = restore_model(model_fn=cnn_model_2d, model_dir=model_dir)

  # Load the data to be tested on. Packed as (ntriggers, nfreq*ntime + 1)
  # where the final column is the class (0 or 1 in this case)
  f = np.load(fn)

  assert (f.shape[-1]-1)==nfreq*ntime, "Input nfreq, ntime are wrong"
  assert twidth <= ntime//2, "twidth should be half ntime or smaller"

  # Make sure eval data is same dtype as trained data
  eval_data = f[::5, :-1].astype(np.float32)
  eval_labels = f[::5, -1].astype(np.int32)

  # Select only central 2*twidth of image
  eval_data = eval_data.reshape(-1, nfreq, ntime)[..., ntime//2-twidth:ntime//2+twidth]

  print("Classifying %d triggers" % len(eval_data))

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
      "precision":
          learn.MetricSpec(
              metric_fn=tf.metrics.precision, prediction_key="classes"),
      "false_negatives":
          learn.MetricSpec(
              metric_fn=tf.metrics.false_negatives, prediction_key="classes"),
      "recall":
          learn.MetricSpec(
              metric_fn=tf.metrics.recall, prediction_key="classes"),
  }

  # Evaluate the model and print results
  eval_results = clf.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)

  label, prob = get_predictions(eval_data, clf)

  return eval_results, label, prob, eval_data, eval_labels

def time_classification(data, clf):
  import time 

  t0 = time.time()
  label, prob = get_predictions(data, clf)
  return time.time() - t0

def time_parameters():
  import glob

  fl = glob.glob('/tmp/frb_convnet_model*')
  fl.sort()
  nf = 16

  W, T = [], []

  for ff in fl:
      clf = restore_model(model_fn=cnn_model_2d, model_dir=ff)
      w=int(ff.split('/')[-1].split('h')[-1])
      W.append(w)
      t=time_classification(f[:1, :nf*2*w].reshape(-1, nf, 2*w).astype(float32), clf)
      T.append(t)

  return np.array(W), np.array(T)

def loop():
  import os

  widths = [16, 32, 80, 120]
  nfilters = [8, 16, 24, 32, 40, 48, 52, 64]
  EV = []
  D = []

  model_dir="./model/frb_convnet_model2"

  for ii in widths:
    for nf in nfilters:
      model_dir_ = model_dir + '_width%d_nf%d/' % (ii, nf)
      global nfilt2
      nfilt2 = nf

      pred, prob, ev, clf, d, p, e = run_cnn_2d(fn, nfreq=16, \
            ntime=250, train_size=0.5, plot=False, twidth=ii, \
            model_dir=model_dir_)

      fout = model_dir_ + '/eval.txt'
      save_results(fout, ev)

    #EV.append(ev)
    #D.append(d[p!=e])

  return EV, widths, D


def tfun(f):
  nfreq, ntime = 16, 250
  eval_data = f[:, :-1]
  eval_data = eval_data.astype(np.float32)  
  eval_data = eval_data.reshape(-1, nfreq, 250)[:, :, ntime//2-32:ntime//2+32]
  pl, prob = get_predictions(eval_data[:], clf)
  indy = np.where(np.array(pl)==1)[0]
  print(indy)
  f = np.delete(f, indy[0], axis=0)
  return f, indy, pl


def gen_1d_timestream(n_batches=1000, ntime=1000, fnout='./data/data_1d.npy', sigma=10.):

  # Create gaussian time streams
  data = np.random.normal(0, 1, n_batches*ntime)
  data = data.reshape(n_batches, ntime).astype(np.float32)

  # Add single spike to central pixel of every second timestamp
  data[::2][:, ntime//2] = 10.0
  labels = np.zeros([n_batches]).astype(np.int32)
  labels[::2] = 1

  reader.write_data(data, labels, fname=fnout)


if __name__=='__main__':

  fn = './data/_data_nt250_nf16_dm0_snr15.npy'

  global nfilt2
  nfilt2 = 64

  if len(sys.argv) > 1:
    fn = sys.argv[1]

  os.system('rm -rf ./model/tf_models/')

  pred, prob, ev, clf, d, p, e = run_cnn_2d(fn, nfreq=16, \
           ntime=250, train_size=0.75, plot=True, twidth=32, \
           model_dir='./model/tf_models/')

  print(ev)

#  loop()









