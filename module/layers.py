# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf
import numpy as np
import math

from tensorflow.contrib.rnn import MultiRNNCell
from tensorflow.contrib.rnn import RNNCell

from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import clip_ops

from functools import reduce
from operator import mul
import sys
sys.path.append('./')

import tensorflow.contrib.microsoft as mstf
'''
This file is from https://github.com/NLPLearn/QANet.git
Some functions are taken directly from Tensor2Tensor Library:
https://github.com/tensorflow/tensor2tensor/
and BiDAF repository:
https://github.com/allenai/bi-att-flow
'''

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

def glu(x):
    """Gated Linear Units from https://arxiv.org/pdf/1612.08083.pdf"""
    x, x_h = tf.split(x, 2, axis = -1)
    return tf.sigmoid(x) * x_h

def noam_norm(x, epsilon=1.0, scope=None, reuse=None):
    """One version of layer normalization."""
    with tf.name_scope(scope, default_name="noam_norm", values=[x]):
        shape = x.get_shape()
        ndims = len(shape)
        return tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(tf.to_float(shape[-1]))

def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias

def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer = regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer = regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

norm_fn = layer_norm#tf.contrib.layers.layer_norm #tf.contrib.layers.layer_norm or noam_norm

def highway(x, size = None, activation = None,
            num_layers = 2, scope = "highway", dropout = 0.0, reuse = None):
    with tf.variable_scope(scope, reuse):
        if size is None:
            size = x.shape.as_list()[-1]
        else:
            x = conv(x, size, name = "input_projection", reuse = reuse)
        for i in range(num_layers):
            T = conv(x, size, bias = True, activation = tf.sigmoid,
                     name = "gate_%d"%i, reuse = reuse)
            H = conv(x, size, bias = True, activation = activation,
                     name = "activation_%d"%i, reuse = reuse)
            H = tf.nn.dropout(H, 1.0 - dropout)
            x = H * T + x * (1.0 - T)
        return x

def layer_dropout(inputs, residual, dropout):
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)

def residual_block(inputs, num_blocks, num_conv_layers, kernel_size, mask = None,
                   num_filters = 128, input_projection = False, num_heads = 8,
                   seq_len = None, scope = "res_block", is_training = True,
                   reuse = None, bias = True, dropout = 0.0):
    with tf.variable_scope(scope, reuse = reuse):
        if input_projection:
            inputs = conv(inputs, num_filters, name = "input_projection", reuse = reuse)
        outputs = inputs
        sublayer = 1
        total_sublayers = (num_conv_layers + 2) * num_blocks
        for i in range(num_blocks):
            outputs = add_timing_signal_1d(outputs)
            outputs, sublayer = conv_block(outputs, num_conv_layers, kernel_size, num_filters,
                seq_len = seq_len, scope = "encoder_block_%d"%i,reuse = reuse, bias = bias,
                dropout = dropout, sublayers = (sublayer, total_sublayers))
            outputs, sublayer = self_attention_block(outputs, num_filters, seq_len, mask = mask, num_heads = num_heads,
                scope = "self_attention_layers%d"%i, reuse = reuse, is_training = is_training,
                bias = bias, dropout = dropout, sublayers = (sublayer, total_sublayers))
        return outputs

def conv_block(inputs, num_conv_layers, kernel_size, num_filters,
               seq_len = None, scope = "conv_block", is_training = True,
               reuse = None, bias = True, dropout = 0.0, sublayers = (1, 1)):
    with tf.variable_scope(scope, reuse = reuse):
        outputs = tf.expand_dims(inputs,2)
        l, L = sublayers
        for i in range(num_conv_layers):
            residual = outputs
            outputs = norm_fn(outputs, scope = "layer_norm_%d"%i, reuse = reuse)
            if (i) % 2 == 0:
                outputs = tf.nn.dropout(outputs, 1.0 - dropout)
            outputs = depthwise_separable_convolution(outputs,
                kernel_size = (kernel_size, 1), num_filters = num_filters,
                scope = "depthwise_conv_layers_%d"%i, is_training = is_training, reuse = reuse)
            outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
            l += 1
        return tf.squeeze(outputs,2), l

def self_attention_block(inputs, num_filters, seq_len, mask = None, num_heads = 8,
                         scope = "self_attention_ffn", reuse = None, is_training = True,
                         bias = True, dropout = 0.0, sublayers = (1, 1)):
    with tf.variable_scope(scope, reuse = reuse):
        l, L = sublayers
        # Self attention
        outputs = norm_fn(inputs, scope = "layer_norm_1", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = multihead_attention(outputs, num_filters,
            num_heads = num_heads, seq_len = seq_len, reuse = reuse,
            mask = mask, is_training = is_training, bias = bias, dropout = dropout)
        residual = layer_dropout(outputs, inputs, dropout * float(l) / L)
        l += 1
        # Feed-forward
        outputs = norm_fn(residual, scope = "layer_norm_2", reuse = reuse)
        outputs = tf.nn.dropout(outputs, 1.0 - dropout)
        outputs = conv(outputs, num_filters, True, tf.nn.relu, name = "FFN_1", reuse = reuse)
        outputs = conv(outputs, num_filters, True, None, name = "FFN_2", reuse = reuse)
        outputs = layer_dropout(outputs, residual, dropout * float(l) / L)
        l += 1
        return outputs, l

def multihead_attention(queries, units, num_heads,
                        memory = None,
                        seq_len = None,
                        scope = "Multi_Head_Attention",
                        reuse = None,
                        mask = None,
                        is_training = True,
                        bias = True,
                        dropout = 0.0):
    with tf.variable_scope(scope, reuse = reuse):
        # Self attention
        if memory is None:
            memory = queries

        memory = conv(memory, 2 * units, name = "memory_projection", reuse = reuse)
        query = conv(queries, units, name = "query_projection", reuse = reuse)
        Q = split_last_dimension(query, num_heads)
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory,2,axis = 2)]

        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head**-0.5
        x = dot_product_attention(Q,K,V,
                                  bias = bias,
                                  seq_len = seq_len,
                                  mask = mask,
                                  is_training = is_training,
                                  scope = "dot_product_attention",
                                  reuse = reuse, dropout = dropout)
        return combine_last_two_dimensions(tf.transpose(x,[0,2,1,3]))

def conv(inputs, output_size, bias = None, activation = None, kernel_size = 1, name = "conv", reuse = None):
    with tf.variable_scope(name, reuse = reuse):
        shapes = inputs.shape.as_list()
        if len(shapes) > 4:
            raise NotImplementedError
        elif len(shapes) == 4:
            filter_shape = [1,kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,1,output_size]
            strides = [1,1,1,1]
        else:
            filter_shape = [kernel_size,shapes[-1],output_size]
            bias_shape = [1,1,output_size]
            strides = 1
        conv_func = tf.nn.conv1d if len(shapes) == 3 else tf.nn.conv2d
        kernel_ = tf.get_variable("kernel_",
                        filter_shape,
                        dtype = tf.float32,
                        regularizer=regularizer,
                        initializer = initializer_relu() if activation is not None else initializer())
        #outputs = conv_func(inputs, kernel_, strides, "VALID")
        outputs = conv_func(inputs, kernel_, strides, "SAME")
        if bias:
            outputs += tf.get_variable("bias_",
                        bias_shape,
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
        if activation is not None:
            return activation(outputs)
        else:
            return outputs

def mask_logits(inputs, mask, mask_value = -1e30):
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope = "depthwise_separable_convolution",
                                    bias = True, is_training = True, reuse = None):
    with tf.variable_scope(scope, reuse = reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable("depthwise_filter",
                                        (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = initializer_relu())
        pointwise_filter = tf.get_variable("pointwise_filter",
                                        (1,1,shapes[-1],num_filters),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = initializer_relu())
        outputs = tf.nn.separable_conv2d(inputs,
                                        depthwise_filter,
                                        pointwise_filter,
                                        strides = (1,1,1,1),
                                        padding = "SAME")
        if bias:
            b = tf.get_variable("bias",
                    outputs.shape[-1],
                    regularizer=regularizer,
                    initializer = tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs

def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
    x: a Tensor with shape [..., m]
    n: an integer.
    Returns:
    a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret,[0,2,1,3])

def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          seq_len = None,
                          mask = None,
                          is_training = True,
                          scope=None,
                          reuse = None,
                          dropout = 0.0):
    """dot-product attention.
    Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    is_training: a bool of training
    scope: an optional string
    Returns:
    A Tensor.
    """
    with tf.variable_scope(scope, default_name="dot_product_attention", reuse = reuse):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias:
            b = tf.get_variable("bias",
                    logits.shape[-1],
                    regularizer=regularizer,
                    initializer = tf.zeros_initializer())
            logits += b
        if mask is not None:
            ############Amanda
            #print(logits.shape)
            #shapes = [x  if x != None else -1 for x in logits.shape.as_list()]
            #print(shapes)
            #shapes = [x if x != None else -1 for x in tf.shape(logits).as_list()]
            #mask = tf.reshape(mask, [shapes[0],1,1,shapes[-1]])
            mask = tf.reshape(mask, [tf.shape(logits)[0],1,1,-1])
            
            #####################################################################
            logits = mask_logits(logits, mask)
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout)
        return tf.matmul(weights, v)

def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
    x: a Tensor with shape [..., a, b]
    Returns:
    a Tensor with shape [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal

def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal

def ndim(x):
    """Copied from keras==2.0.6
    Returns the number of axes in a tensor, as an integer.
    # Arguments
        x: Tensor or variable.
    # Returns
        Integer (scalar), number of axes.
    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(inputs)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None

def dot(x, y):
    """Modified from keras==2.0.6
    Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)
    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.
    # Returns
        A tensor, dot product of `x` and `y`.
    """
    if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out

def batch_dot(x, y, axes=None):
    """Copy from keras==2.0.6
    Batchwise dot product.
    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.
    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.
    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = ndim(x)
    y_ndim = ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if ndim(x) == 2 and ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == ndim(x) - 1 else True
            adj_y = True if axes[1] == ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out

def optimized_trilinear_for_attention(args, c_maxlen, q_maxlen, input_keep_prob=1.0,
    scope='efficient_trilinear',
    bias_initializer=tf.zeros_initializer(),
    kernel_initializer=initializer()):
    assert len(args) == 2, "just use for computing attention with two input"
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("`args` must be 3 dims (batch_size, len, dimension)")
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("the last dimension of `args` must equal")
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    droped_args = [tf.nn.dropout(arg, input_keep_prob) for arg in args]
    with tf.variable_scope(scope):
        weights4arg0 = tf.get_variable(
            "linear_kernel4arg0", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4arg1 = tf.get_variable(
            "linear_kernel4arg1", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4mlu = tf.get_variable(
            "linear_kernel4mul", [1, 1, arg_size],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        biases = tf.get_variable(
            "linear_bias", [1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=bias_initializer)
        subres0 = tf.tile(dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])
        subres1 = tf.tile(tf.transpose(dot(droped_args[1], weights4arg1), perm=(0, 2, 1)), [1, c_maxlen, 1])
        subres2 = batch_dot(droped_args[0] * weights4mlu, tf.transpose(droped_args[1], perm=(0, 2, 1)))
        res = subres0 + subres1 + subres2
        nn_ops.bias_add(res, biases)
        return res

def trilinear(args,
            output_size = 1,
            bias = True,
            squeeze=False,
            wd=0.0,
            input_keep_prob= 1.0,
            scope = "trilinear"):
    with tf.variable_scope(scope):
        flat_args = [flatten(arg, 1) for arg in args]
        flat_args = [tf.nn.dropout(arg, input_keep_prob) for arg in flat_args]
        flat_out = _linear(flat_args, output_size, bias, scope=scope)
        out = reconstruct(flat_out, args[0], 1)
        return tf.squeeze(out, -1)

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat

def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

def _linear(args,
            output_size,
            bias,
            bias_initializer=tf.zeros_initializer(),
            scope = None,
            kernel_initializer=initializer(),
            reuse = None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]
  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope, reuse = reuse) as outer_scope:
    weights = tf.get_variable(
        "linear_kernel", [total_arg_size, output_size],
        dtype=dtype,
        regularizer=regularizer,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable(
          "linear_bias", [output_size],
          dtype=dtype,
          regularizer=regularizer,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)

def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))

#Newly added
def convert2vec(ids_str):
    ids_split = tf.string_split(ids_str)
    ids_tensor = tf.SparseTensor(indices = ids_split.indices, values = tf.string_to_number(ids_split.values,out_type=tf.int32), dense_shape = ids_split.dense_shape)
    return tf.sparse_tensor_to_dense(ids_tensor,default_value=-1)
def mstf_xletter_maxpool(text, model_prefix, op_dict=None, xletter_cnt = None, win_size=None, dim_xletter_emb=None):
    with tf.variable_scope("xletter_layer", reuse=tf.AUTO_REUSE):
        xletter_emb = tf.get_variable(name='xletter_emb_' + model_prefix, shape = [xletter_cnt * win_size, dim_xletter_emb])
        indices, ids, values, offsets = mstf.dssm_xletter(input=text, win_size=win_size, dict_handle=op_dict)
        vec, _ = mstf.dssm_conv(input_indices=indices, input_ids=ids, input_values=values,input_offsets=offsets,weight=xletter_emb,max_pooling=True)
    return vec
def xletter_feature_extractor(text,model_prefix,input_mode, op_dict=None,xletter_cnt=None,win_size=None,dim_xletter_emb=None):
    with tf.variable_scope("xletter_layer", reuse=tf.AUTO_REUSE):
        if input_mode=='mstf':
            xletter_emb = tf.get_variable(name='xletter_emb_' + model_prefix, shape = [xletter_cnt * win_size, dim_xletter_emb])
            indices, ids, values, offsets = mstf.dssm_xletter(input=text, win_size=win_size, dict_handle=op_dict)
            offsets_to_dense = tf.segment_sum(tf.ones_like(offsets), offsets)
            batch_id = tf.cumsum(offsets_to_dense[:-1])
            index_tensor = tf.concat([tf.expand_dims(batch_id,axis=-1), tf.expand_dims(indices,axis=-1)],axis=-1)
            value_tensor = ids
            dense_shape = tf.concat([tf.shape(offsets),tf.expand_dims(tf.reduce_max(indices) + 1,axis=-1)],axis=0)
            text_tensor = tf.SparseTensor(indices=tf.cast(index_tensor,tf.int64), values = value_tensor, dense_shape=tf.cast(dense_shape,tf.int64))
            #conv
            text_tensor = tf.sparse_reshape(text_tensor,[-1])
            text_tensor,text_mask = tf.sparse_fill_empty_rows(text_tensor,0)
            text_vecs = tf.nn.embedding_lookup_sparse(xletter_emb,text_tensor,None,combiner='sum')
            text_vecs = tf.where(~text_mask, text_vecs, tf.zeros_like(text_vecs))
            text_vecs = tf.reshape(text_vecs,[-1,tf.reduce_max(indices) + 1,dim_xletter_emb])
            step_mask = ~tf.equal(tf.reduce_sum(text_vecs,axis=2),0)
            sequence_length = tf.cast(tf.count_nonzero(step_mask,axis=1),tf.int32)
        elif input_mode=='pyfunc':
            query_split = tf.string_split(text,';')
            term_split = tf.string_split(query_split.values,',')
            xletter_tensor_indices = tf.transpose(tf.stack([tf.gather(query_split.indices[:,0],term_split.indices[:,0]),tf.gather(query_split.indices[:,1],term_split.indices[:,0])]))
            xletter_tensor = tf.SparseTensor(indices = xletter_tensor_indices, values = tf.string_to_number(term_split.values,out_type=tf.int32), dense_shape = query_split.dense_shape)
            xletter_emb = tf.get_variable(name='xletter_emb_' + model_prefix, shape = [xletter_cnt * win_size, dim_xletter_emb])
            xletter_tensor_reshape = tf.sparse_reshape(xletter_tensor,[-1])
            xletter_tensor,text_mask = tf.sparse_fill_empty_rows(xletter_tensor_reshape,0)
            xletter_vecs = tf.nn.embedding_lookup_sparse(xletter_emb, xletter_tensor, None, combiner='sum')
            xletter_vecs = tf.where(~text_mask, xletter_vecs, tf.zeros_like(xletter_vecs))
            text_vecs = tf.reshape(xletter_vecs, shape=tf.stack([-1,tf.reduce_max(query_split.indices[:,1])+1,dim_xletter_emb]))
            step_mask = ~tf.equal(tf.reduce_sum(text_vecs,axis=2),0)
            sequence_length = tf.cast(tf.count_nonzero(step_mask,axis=1),tf.int32)
        elif input_mode=='pyfunc_batch':
            indices, values, dense_shape = tf.py_func(op_dict.batch_xletter_extractor,[text],[tf.int64,tf.int32,tf.int64])
            xletter_tensor = tf.SparseTensor(indices = indices, values = values, dense_shape = dense_shape)
            xletter_emb = tf.get_variable(name='xletter_emb_' + model_prefix, shape = [xletter_cnt * win_size, dim_xletter_emb])
            xletter_tensor_reshape = tf.sparse_reshape(xletter_tensor,[-1])
            xletter_tensor,text_mask = tf.sparse_fill_empty_rows(xletter_tensor_reshape,0)
            xletter_vecs = tf.nn.embedding_lookup_sparse(xletter_emb, xletter_tensor, None, combiner='sum')
            xletter_vecs = tf.where(~text_mask, xletter_vecs, tf.zeros_like(xletter_vecs))
            text_vecs = tf.reshape(xletter_vecs, shape=tf.stack([-1,dense_shape[1],dim_xletter_emb]))
            step_mask = ~tf.equal(tf.reduce_sum(text_vecs,axis=2),0)
            sequence_length = tf.cast(tf.count_nonzero(step_mask,axis=1),tf.int32)
        else:
            NotImplementedError
    return text_vecs, step_mask, sequence_length

def lookup_emb(text_tensor, text_padding, embedding_weight, dim_output):
        #conv
    text_tensor = tf.sparse_reshape(text_tensor,[-1])
    text_tensor,text_mask = tf.sparse_fill_empty_rows(text_tensor,0)
    text_vecs = tf.nn.embedding_lookup_sparse(embedding_weight,text_tensor,None,combiner='sum')
    text_vecs = tf.where(~text_mask, text_vecs, tf.zeros_like(text_vecs))
    text_vecs = tf.reshape(text_vecs,shape=tf.stack([-1,text_padding,dim_output]))
    step_mask = ~tf.equal(tf.reduce_sum(text_vecs,axis=2),0)
    sequence_length = tf.cast(tf.count_nonzero(step_mask,axis=1),tf.int32)
    return text_vecs, step_mask, sequence_length

def term_emb_extract(text,w2v_vocab,w2v_emb,dim_w2v_emb,add_terminator=False):
    text_tensor = tf.string_split(text)
    if add_terminator:
        dense_shape = tf.stack([text_tensor.dense_shape[0],text_tensor.dense_shape[1] + 1])
    else:
        dense_shape = tf.stack([text_tensor.dense_shape[0],text_tensor.dense_shape[1]])
    text_tensor_expand = tf.SparseTensor(indices = text_tensor.indices, values = w2v_vocab.lookup(text_tensor.values), dense_shape=dense_shape)
    text_padding = text_tensor_expand.dense_shape[1]
    text_vecs, step_mask, sequence_length = lookup_emb(text_tensor_expand, text_padding, w2v_emb, dim_w2v_emb)
    if add_terminator:
        return text_vecs, step_mask, sequence_length,tf.sparse_tensor_to_dense(text_tensor_expand)
    return text_vecs, step_mask, sequence_length

def term_id_extract(text,w2v_vocab):
    text_tensor = tf.string_split(text)
    dense_shape = tf.stack([text_tensor.dense_shape[0],text_tensor.dense_shape[1]])
    text_tensor_expand = tf.SparseTensor(indices = text_tensor.indices, values = w2v_vocab.lookup(text_tensor.values), dense_shape=dense_shape)
    text_padding = text_tensor_expand.dense_shape[1]
    text_id = tf.sparse_tensor_to_dense(text_tensor_expand)
    text_mask = ~tf.equal(text_id,0)
    return text_id, text_mask

    
def char_emb_extract(text, c2v_vocab, c2v_emb, dim_c2v_emb, dim_hidden):
    #[batch_size, textLen, wordLen, dim_c2v_emb]
    char_emb, char_mask, char_len, text_mask, text_len = lookup_char_emb(text, c2v_vocab, c2v_emb, dim_c2v_emb)
    #[batch_size*textLen, wordLen, dim_c2v_emb]
    ch_emb = tf.reshape(char_emb, shape=[-1,tf.shape(char_emb)[2],dim_c2v_emb])
    #print("ch_emb01",ch_emb)
    #return char_emb,None,tf.shape(ch_emb_reshape)
    #return tf.shape(char_emb), tf.shape(ch_emb_), None
    #Conv
    ch_emb = conv(ch_emb, dim_hidden, bias = True, activation=tf.nn.relu, kernel_size=5, name='char_conv',reuse=tf.AUTO_REUSE)
    #print("ch_emb02", ch_emb)
    #return char_emb, None, tf.shape(ch_emb)
    #return tf.shape(ch_emb_conv), tf.shape(ch_emb), None
    ch_emb = tf.reshape(ch_emb, shape=[tf.shape(text)[0],-1,tf.shape(ch_emb)[-2],dim_hidden])
    #return char_emb, None, tf.shape(ch_emb)
    #return tf.shape(char_emb), tf.shape(char_mask), None
    ch_emb = mask_maxpool(ch_emb, char_mask)
    #return char_emb, text_mask, tf.shape(ch_emb)
    return ch_emb, text_mask, text_len

def lookup_char_emb(text,c2v_vocab, c2v_emb, dim_c2v_emb):
    str_tensor = tf.string_split(text)
    str_split = tf.sparse_reshape(str_tensor,[-1])
    str_split,text_mask = tf.sparse_fill_empty_rows(str_split,"")
    #return str_split
    #str_split = tf.sparse_tensor_to_dense(str_split,default_value="")
    #char_split = tf.string_split(str_split.values,'')
    char_split = tf.string_split(str_split.values,'')
    #return char_split
    #return tf.SparseTensor(indices=tf.stack([str_split.indices,res.indices],axis=1),values = res.values, dense_shape=tf.stack([str_split.dense_shape[0],str_split.dense_shape[1],res.dense_shape[0], res.dense_shape[1]]))
    #return char_split
    #char_tensor_indices = tf.transpose(tf.stack([tf.gather(str_split.indices[:,0],char_split.indices[:,0]),tf.gather(str_split.indices[:,1],char_split.indices[:,0])]))
    char_tensor = tf.SparseTensor(indices = char_split.indices, values = c2v_vocab.lookup(char_split.values), dense_shape = char_split.dense_shape)
    #return char_tensor
    #char_tensor = tf.SparseTensor(indices = char_split.indices, values = char_dict.lookup(char_split.values), dense_shape = char_split.dense_shape)
    char_tensor_reshape = tf.sparse_reshape(char_tensor,[-1])
    char_tensor,term_mask = tf.sparse_fill_empty_rows(char_tensor_reshape,0)
    #return char_tensor
    char_vecs = tf.nn.embedding_lookup_sparse(c2v_emb, char_tensor, None, combiner='sum')
    char_vecs = tf.where(~term_mask, char_vecs, tf.zeros_like(char_vecs))
    #return char_vecs
    term_char_vecs = tf.reshape(char_vecs, shape = tf.stack([tf.shape(text)[0],tf.cast(tf.reduce_max(str_tensor.indices[:,1])+1,tf.int32),-1,tf.shape(char_vecs)[-1]]))
    term_char_mask_tmp = tf.reduce_sum(term_char_vecs,axis=-1)
    term_char_mask = ~tf.equal(term_char_mask_tmp,0)
    term_char_len = tf.cast(tf.count_nonzero(term_char_mask,axis=-1),tf.int32)
    text_mask = ~tf.equal(tf.reduce_sum(term_char_mask_tmp,axis=-1),0)
    text_len = tf.cast(tf.count_nonzero(text_mask,axis=-1),tf.int32)
    return term_char_vecs, term_char_mask, term_char_len, text_mask, text_len

def sentence_emb(vecs, text_mask, text_len, dim_hidden, activation, use_bias=True):
    vec = mask_maxpool(vecs, text_mask)
    return tf.layers.dense(vec, dim_hidden, activation=activation,reuse=tf.AUTO_REUSE,use_bias=use_bias)

def mask_maxpool(text_vecs, step_mask):
    step_mask = tf.where(~step_mask ,-math.inf*tf.ones_like(step_mask,dtype=tf.float32),tf.zeros_like(step_mask,dtype=tf.float32))
    vecs_4maxpool = text_vecs + tf.expand_dims(step_mask,axis=-1)
    maxpool = tf.reduce_max(vecs_4maxpool, axis=-2)
    maxpool = tf.where(tf.is_finite(maxpool), maxpool, tf.zeros_like(maxpool))
    return maxpool

#text_vecs: [batch_size, seq_len, vector]
#step_mask: [batch_size, seq_len]
def mask_avgpool(text_vecs, step_mask):
    #seq_len = tf.count_nonzero(step_mask,axis=-1)
    #text_mask = tf.tile(tf.expand_dims(step_mask,-1),[1,1,tf.shape(text_vecs)[-1]])
    #text_vecs = tf.where(text_mask, text_vecs, tf.zeros_like(text_vecs))
    #text_vec = tf.reduce_sum(text_vecs,axis=1) / seq_len
    seq_len = tf.cast(tf.count_nonzero(step_mask,axis=-1),tf.float32)
    text_mask = tf.tile(tf.expand_dims(step_mask,-1),[1,1,tf.shape(text_vecs)[-1]])
    text_vecs = tf.where(text_mask, text_vecs, tf.zeros_like(text_vecs))
    seq_div = tf.where(tf.equal(seq_len,0),tf.ones_like(seq_len),seq_len)
    seq_div = tf.tile(tf.expand_dims(seq_div,-1),[1,tf.shape(text_vecs)[-1]])
    text_vec = tf.reduce_sum(text_vecs,axis=1) / seq_div
    return text_vec

if __name__ == '__main__': 
    q = tf.placeholder(tf.string)
    num = (convert2vec(q))

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        #sess.run(tf.local_variables_initializer())
        #sess.run(tf.tables_initializer())
        fea = sess.run([num], feed_dict={q:["0 0 0 0 1 1 1","0 0 1 1"]})
        print(fea)

