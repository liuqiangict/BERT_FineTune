import collections
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import control_flow_ops
class ConstrainedDecoderOutput(
        collections.namedtuple("ConstrainedDecoderOutput",("rnn_output","sample_id"))):
    pass

class ConstrainedDecoder(decoder.Decoder):
    def __init__(self, cell, initial_state, embedding, start_tokens, end_token, constrained_matrix, output_layer = None):
        self._cell = cell
        self._initial_state = initial_state
        if output_layer is not None and not isinstance(output_layer, layers_base.Layer):
            raise TypeError("output_layer must be a Layer, received: %s" %type(output_layer))
        self._output_layer = output_layer
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (lambda ids: embedding_ops.embedding_lookup(embedding,ids))
        self._start_tokens = ops.convert_to_tensor(start_tokens, dtype=dtypes.int32, name="start_tokens")
        self._end_token = ops.convert_to_tensor(end_token, dtype=dtypes.int32, name="end_token")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = array_ops.size(start_tokens)
        if self._end_token.get_shape().ndims != 0:
            raise ValueError("end_token must be a scalar")
        self._start_inputs = self._embedding_fn(self._start_tokens)
        self._constrained_matrix = constrained_matrix
    
    @property
    def batch_size(self):
        return self._batch_size

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size
        else:
            output_shape_with_unknown_batch = nest.map_structure( lambda s: tensor_shape.TensorShape([None]).concatenate(s), size)
            layer_output_shape = self._output_layer._compute_output_shape(output_shape_with_unknown_batch)
        return nest.map_structure(lambda s: s[1:], layer_output_shape)
    
    @property
    def output_size(self):
        return ConstrainedDecoderOutput(rnn_output=self._rnn_output_size(), sample_id=tensor_shape.TensorShape([]))

    @property
    def output_dtype(self):
        dtype = nest.flatten(self._initial_state)[0].dtype
        return ConstrainedDecoderOutput(nest.map_structure(lambda _:dtype, self._rnn_output_size()), dtypes.int32)

    def initialize(self, name=None):
        """Initialize the decoder.
        Args:
        name: Name scope for any created operations.
        Returns:
        `(finished, first_inputs, initial_state)`.
        """
        #finsished: Flag to indicate if the batch finished decoding
        finished = array_ops.tile([False],[self._batch_size])
        return (finished, self._start_inputs, self._initial_state)

    def step(self, time, inputs, state, name=None):
        """Perform a decoding step.
         Args:
         time: scalar `int32` tensor.
         inputs: A (structure of) input tensors.
         state: A (structure of) state tensors and TensorArrays.
         name: Name scope for any created operations.
         Returns:
         `(outputs, next_state, next_inputs, finished)`.
        """
        #It will automatically input (time+1, next_inputs, finshed) to next step
        with ops.name_scope(name, "ConstrainedDecoderStep",(time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
            #Calculate sample id based on constrained matrix
            #cell_outputs = tf.multiply(cell_outputs, self._constrained_matrix)
            cell_outputs_mask = cell_outputs + self._constrained_matrix
            sample_ids = math_ops.argmax(cell_outputs_mask, axis=-1, output_type=tf.int32)
            finished = math_ops.equal(sample_ids, self._end_token)
            all_finished = math_ops.reduce_all(finished)
            next_inputs = control_flow_ops.cond(all_finished, lambda: self._start_inputs, lambda:self._embedding_fn(sample_ids))
            outputs = ConstrainedDecoderOutput(cell_outputs, sample_ids)
            return (outputs, cell_state, next_inputs, finished)


def generate_constrained_matrix(doc, decoder_dict, decoder_vocab_size):
    constrained_tensor = tf.string_split(doc)
    constrained_ind = tf.cast(tf.stack([constrained_tensor.indices[:,0], decoder_dict.lookup(constrained_tensor.values)],axis=-1),tf.int64)
    constrained_val = tf.zeros_like(constrained_tensor.indices[:,0],dtype=tf.float32)
    constrained_shape=tf.cast(tf.stack([tf.shape(doc)[0],decoder_vocab_size]),dtype=tf.int64)
    fix_matrix = tf.tile(tf.sparse_to_dense([[0,EOS_ID]],[1,decoder_vocab_size],[True],default_value=False, validate_indices=False),[tf.shape(doc)[0],1])
    constrained_matrix = tf.sparse_to_dense(constrained_ind, constrained_shape, constrained_val, default_value=-np.inf, validate_indices=False)
    constrained_matrix = tf.where(fix_matrix, tf.zeros_like(constrained_matrix), constrained_matrix)
