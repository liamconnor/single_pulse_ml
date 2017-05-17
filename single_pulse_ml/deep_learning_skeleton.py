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
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

#print(sess.run(tf.argmax(y, 1), feed_dict={x: mnist.test.images}))

def get_predictions(data, classifier):
  predictions = classifier.predict(data, input_fn=None)

  label, prob = [], []

  while True:
    try:
      pred = predictions.next()
      label.append(pred['classes'])
      prob.append(pred['probabilities'])
    except StopIteration:
      return label, np.array(prob)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
#  input_layer = tf.reshape(features, [-1, 28, 28, 1])
  h, w = int(features.shape[1]), int(features.shape[2])
  input_layer = tf.reshape(features, [-1, h, w, 1])


  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  #  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  # data have been pooled twice; new shape is h//4, w//4, nfeatures
  pool2_flat = tf.reshape(pool2, [-1, h//4 * w//4 * 64])

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

def liamsmain(train_data, train_labels, eval_data, eval_labels):
  # with 1000:
  #{'loss': 0.16229428, 'global_step': 7663, 'accuracy': 0.96799999}

  # Create the Estimator
  mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

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

def dedisp(data, dm, freq=np.linspace(800, 400, 1024)):

    dm_del = 4.148808e3 * dm * (freq**(-2) - 600.0**(-2))
    dt = 512 * 2.56e-6

    A2 = np.zeros_like(data)

    for ii, ff in enumerate(freq):
        dmd = int(round(dm_del[ii] / dt))
        A2[ii] = np.roll(data[ii], -dmd, axis=-1)

    return A2

def dm_delays(dm, freq, f_ref):

    return 4.148808e3 * dm * (freq**(-2) - f_ref**(-2))

def straighten_arr(data):

    sn = []

    for dm in dms:
      d_ = dedisp(data.copy(), dm, freq=linspace(800,400,16))
      sn.append(d_.mean(0).max() / np.std(d_.mean(0)))

    d_ = dedisp(data, dms[np.argmax(sn)], freq=linspace(800,400,16))

    return d_

def run_straightening(dd):
  for ii in range(len(dd)):
    dd_ = dd[ii].reshape(-1, 250)                
    dd[ii] = (straighten_arr(dd_)).reshape(-1)

  return dd

fn = '/Users/connor/training_data_pfFreq.npy'
fn = '/Users/connor/code/machine_learning/single_pulse_ml/single_pulse_ml/full_data_pfFreq_sims.npy'
fn = '/Users/connor/code/machine_learning/single_pulse_ml/single_pulse_ml/full_data_pfFreq_all_sims.npy'

f = np.load(fn)

td, ed, tl, el = train_test_split(f[:, :-1], f[:, -1], train_size=0.75)
#td, tl, ed, el = f[:200, :-1],f[:200, -1],f[200:, :-1],f[200:, -1]
#td, tl, ed, el = f[:1e3, :-1],f[:1e3, -1],f[1e3:, :-1],f[1e3:, -1]
td = td.reshape(-1, 16, 250)[:, :, 125-64:125+64]
#td = td.reshape(-1, 32, 50, 5).mean(-1)
ed = ed.reshape(-1, 16, 250)[:, :, 125-64:125+64]
#ed = ed.reshape(-1, 32, 50, 5).mean(-1)

print(td.shape, ed.shape)
td, tl, ed, el = td.astype(np.float32)[:, :], tl.astype(np.int32), ed.astype(np.float32)[:], el.astype(np.int32)
res, mn = liamsmain(td, tl, ed, el)

# if __name__ == "__main__":
#   if len(sys.argv) < 2:
#     fn = '/Users/connor/training_data_pfFreq.npy'
#   else:
#     fn = sys.argv[1]

#   f = np.load(fn)
#   print(f.shape)

#   td, tl, ed, el = f[:200, :-1],f[:200, -1],f[200:, :-1],f[200:, -1]
# #  td = td.reshape(-1, 32, 250)[:, 4:, 125-14:125+14]
#   td = td.reshape(-1, 32, 50, 5).mean(-1)
# #  ed = ed.reshape(-1, 32, 250)[:, 4:, 125-14:125+14]
#   ed = ed.reshape(-1, 32, 50, 5).mean(-1)

#   print(td.shape, ed.shape)
#   td, tl, ed, el = td.astype(np.float32)[:, :], tl.astype(np.int32), ed.astype(np.float32)[:], el.astype(np.int32)
#   res, mn = liamsmain(td, tl, ed, el)

#  tf.app.run()



# def main(unused_argv):
#   # with 1000:
#   #{'loss': 0.16229428, 'global_step': 7663, 'accuracy': 0.96799999}

#   # Load training and eval data
#   mnist = learn.datasets.load_dataset("mnist")
#   train_data = mnist.train.images  # Returns np.array
#   train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
#   eval_data = mnist.test.images  # Returns np.array
#   eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

#   # Create the Estimator
#   mnist_classifier = learn.Estimator(
#       model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

#   # Set up logging for predictions
#   # Log the values in the "Softmax" tensor with label "probabilities"
#   tensors_to_log = {"probabilities": "softmax_tensor"}
#   logging_hook = tf.train.LoggingTensorHook(
#       tensors=tensors_to_log, every_n_iter=50)

#   # Train the model
#   mnist_classifier.fit(
#       x=train_data,
#       y=train_labels,
#       batch_size=100,
#       steps=20000,
#       monitors=[logging_hook])

#   # Configure the accuracy metric for evaluation
#   metrics = {
#       "accuracy":
#           learn.MetricSpec(
#               metric_fn=tf.metrics.precision, prediction_key="classes"),
#   }

#   # Evaluate the model and print results
#   eval_results = mnist_classifier.evaluate(
#       x=eval_data, y=eval_labels, metrics=metrics)
# #  print(eval_results)




