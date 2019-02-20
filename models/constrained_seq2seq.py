import sys
sys.path.append('.')
import tensorflow as tf
from utils.param import FLAGS
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
from utils.xletter import XletterPreprocessor
from utils.layers import xletter_feature_extractor, term_emb_extract
from models.constrained_decoder import ConstrainedDecoder
from models.constrained_beam_search_decoder import ConstrainedBeamSearchDecoder
import numpy as np
if FLAGS.use_mstf_ops == 1:
    import tensorflow.contrib.microsoft as mstf
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 2
SOS_ID = 1
EOS_ID = 0

def default_init():
        # replica of tf.glorot_uniform_initializer(seed=seed)
    return tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode="FAN_AVG",uniform=True) #seed=seed)

class Seq2Seq():
    def __init__(self):
        #Dictionary initialization
        if FLAGS.use_mstf_ops == 1:
            self.op_dict = mstf.dssm_dict(FLAGS.xletter_dict)
        elif FLAGS.use_mstf_ops == -1:
            self.op_dict = XletterPreprocessor(FLAGS.xletter_dict, FLAGS.xletter_win_size)
        else:
            self.op_dict = None
        #self.decoder_dict = lookup_ops.index_table_from_file(FLAGS.input_previous_model_path + "/" + FLAGS.decoder_vocab_file, default_value=UNK_ID)
        #self.reverse_decoder_dict = lookup_ops.index_to_string_table_from_file(FLAGS.input_previous_model_path + "/" + FLAGS.decoder_vocab_file, default_value=UNK)
        self.decoder_dict = lookup_ops.index_table_from_file(FLAGS.output_model_path + "/" + FLAGS.decoder_vocab_file, default_value=UNK_ID)
        self.reverse_decoder_dict = lookup_ops.index_to_string_table_from_file(FLAGS.output_model_path + "/" + FLAGS.decoder_vocab_file, default_value=UNK)
    def inference(self, input_fields, mode):
        if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
            if FLAGS.use_mstf_ops:
                query, doc = input_fields[0], input_fields[1]
            else:
                query, doc = input_fields[0][0], input_fields[1]
        else:
            if FLAGS.use_mstf_ops:
                query,doc = input_fields[0], None
            else:
                query,doc = input_fields[0][0], None
        if FLAGS.training_with_weight and mode == tf.contrib.learn.ModeKeys.TRAIN:
            weight = tf.string_to_number(input_fields[2])
        else:
            weight = None
        if FLAGS.predict_with_constrain and mode == tf.contrib.learn.ModeKeys.INFER:
            doc = input_fields[1]
        output_state, c_state, source_sequence_length = self.encoder(query)
        logits, sample_id, final_state, target_id, target_sequence_length = self.decoder(output_state, c_state, source_sequence_length, doc, mode)
        if not mode == tf.contrib.learn.ModeKeys.INFER:
            return [logits, target_id, target_sequence_length,weight]
        else:
            return [logits, sample_id, target_sequence_length,weight]


    def encoder(self, query):
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            #q_vecs, sequence_length = self.xletter_feature_extract(query)
            q_vecs, q_mask, sequence_length = xletter_feature_extractor(query, 'Q', self.op_dict, FLAGS.xletter_cnt, FLAGS.xletter_win_size, FLAGS.dim_xletter_emb)
            encoder_cells = [self.build_cell('encoder_' + str(idx) , FLAGS.dim_encoder) for idx in range(0,FLAGS.num_hidden_layers)]
            encoder_cells += [self.build_cell('encoder_'+str(FLAGS.num_hidden_layers), FLAGS.dim_attention)]
            encoder_cell = tf.contrib.rnn.MultiRNNCell(encoder_cells)
            initial_state = encoder_cell.zero_state(tf.shape(q_vecs)[0], dtype=tf.float32)
            output,c_state = tf.nn.dynamic_rnn(encoder_cell, q_vecs, initial_state = initial_state, sequence_length=sequence_length)
        return output, c_state, sequence_length

    def decoder(self, encoder_outputs, encoder_state, encoder_sequence_length, doc, mode):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as decoder_scope:
            decoder_emb = tf.get_variable(name='decoder_emb', shape=[FLAGS.decoder_vocab_size, FLAGS.dim_decoder_emb])
            decoder_cells = [self.build_cell('decoder_'+str(idx), FLAGS.dim_decoder) for idx in range(0,FLAGS.num_hidden_layers)]
            decoder_cells += [self.build_cell('decoder_'+str(FLAGS.num_hidden_layers), FLAGS.dim_attention)]
            decoder_cell = tf.contrib.rnn.MultiRNNCell(decoder_cells)
            output_layer = layers_core.Dense(FLAGS.decoder_vocab_size, name='output_projection',kernel_initializer=default_init())
            ##Newly Add
            #attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
            #alignment_history = False
            #cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = FLAGS.dim_attention, alignment_history=alignment_history, output_attention = True, name='attention')
            #initial_state = cell.zero_state(tf.shape(encoder_sequence_length)[0],tf.float32).clone(cell_state = encoder_state)
            #######################
            if not mode == tf.contrib.learn.ModeKeys.INFER:
                target_input = doc
               #target_emb,step_mask,sequence_length,target_id = self.term_emb_extract(target_input, decoder_emb)
                attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
                alignment_history = False
                cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = FLAGS.dim_attention, alignment_history=alignment_history, output_attention = True, name='attention')
                #cell = decoder_cell
                initial_state = cell.zero_state(tf.shape(encoder_sequence_length)[0],tf.float32).clone(cell_state = encoder_state)
                #initial_state = encoder_state
                #output_layer = layers_core.Dense(FLAGS.decoder_vocab_size, name='output_projection', kernel_initializer = default_init())
               ############
                target_emb,step_mask,sequence_length,target_id = term_emb_extract(target_input, self.decoder_dict, decoder_emb, FLAGS.dim_decoder_emb,add_terminator=True)
                sos_emb = tf.tile(tf.nn.embedding_lookup(decoder_emb,SOS_ID),tf.stack([tf.shape(sequence_length)[0]]))
                sos_emb = tf.reshape(sos_emb,tf.stack([tf.shape(sequence_length)[0],1,FLAGS.dim_decoder_emb]))
                target_emb = tf.concat([sos_emb,target_emb],axis=1)
                sequence_length = sequence_length + 1
                helper = tf.contrib.seq2seq.TrainingHelper(target_emb,sequence_length,time_major=False)
               #Decoder
               #my_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state)
                outputs, c_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, output_time_major = False,impute_finished = False,scope=decoder_scope)
               #??
                sample_id = outputs.sample_id
                logits = output_layer(outputs.rnn_output)
            else:
                beam_width = FLAGS.beam_width
                start_tokens = tf.fill([tf.shape(encoder_outputs)[0]],SOS_ID)
                end_token = EOS_ID
                if FLAGS.predict_with_constrain:
                    constrained_tensor = tf.string_split(doc)
                    constrained_ind = tf.cast(tf.stack([constrained_tensor.indices[:,0], self.decoder_dict.lookup(constrained_tensor.values)],axis=-1),tf.int64)
                    constrained_val = tf.zeros_like(constrained_tensor.indices[:,0],dtype=tf.float32)
                    constrained_shape=tf.cast(tf.stack([tf.shape(doc)[0],FLAGS.decoder_vocab_size]),dtype=tf.int64)
                    fix_matrix = tf.tile(tf.sparse_to_dense([[0,EOS_ID]],[1,FLAGS.decoder_vocab_size],[True],default_value=False, validate_indices=False),[tf.shape(doc)[0],1])
                    #constrained_sparse_matrix = tf.SparseTensor(indices = constrained_ind, values = constrained_val, dense_shape=constrained_shape)#,validate_indices=False)
                    constrained_matrix = tf.sparse_to_dense(constrained_ind, constrained_shape, constrained_val, default_value=-np.inf, validate_indices=False)
                    constrained_matrix = tf.where(fix_matrix, tf.zeros_like(constrained_matrix), constrained_matrix)
                    #constrained_matrix += fix_matrix
                    #constrained_matrix = tf.sparse_tensor_to_dense(constrained_sparse_matrix)
                else:
                    constrained_matrix = None
                #self.tmp = constrained_matrix
                if beam_width > 0:
                    encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
                    encoder_sequence_length = tf.contrib.seq2seq.tile_batch(encoder_sequence_length, multiplier=beam_width)
                    encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
                    true_batch_size = tf.shape(encoder_sequence_length)[0]
                    attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
                    cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = FLAGS.dim_attention, alignment_history=False, output_attention = True, name='attention')
                    #cell = decoder_cell
                    initial_state = cell.zero_state(batch_size=true_batch_size, dtype=tf.float32)
                    initial_state = initial_state.clone(cell_state=encoder_state)
                    #initial_state = encoder_state
                    #if FLAGS.predict_with_constrain:
                    my_decoder = ConstrainedBeamSearchDecoder(
                        cell = cell,
                        embedding=decoder_emb,
                        start_tokens = start_tokens,
                        end_token = end_token,
                        initial_state=initial_state,
                        beam_width=beam_width,
                        constrained_matrix = constrained_matrix,
                        output_layer = output_layer,
                        length_penalty_weight=FLAGS.length_penalty_weight,
                        coverage_penalty_weight = FLAGS.coverage_penalty_weight
                        )
                    #else:
                    #    my_decoder = NoConstrainedBeamSearchDecoder(
                    #        cell = cell,
                    #        embedding=decoder_emb,
                    #        start_tokens = start_tokens,
                    #        end_token = end_token,
                    #        initial_state=initial_state,
                    #        beam_width=beam_width,
                    #        output_layer = output_layer,
                    #        length_penalty_weight=FLAGS.length_penalty_weight,
                    #        coverage_penalty_weight=FLAGS.coverage_penalty_weight
                    #        )
                        #my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        #        cell = cell,
                        #        embedding = decoder_emb, 
                        #        start_tokens = start_tokens,
                        #        end_token = end_token,
                        #        initial_state = initial_state,
                        #        beam_width = beam_width,
                        #        output_layer = output_layer,
                        #        length_penalty_weight=0.0)


                #else:
                #   #attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
                #   #alignment_history = False
                #   #cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = FLAGS.dim_attention, alignment_history=alignment_history, output_attention = True, name='attention')
                #   cell = decoder_cell
                #   initial_state = encoder_state
                #   #initial_state = cell.zero_state(tf.shape(encoder_sequence_length)[0],tf.float32).clone(cell_state = encoder_state)
                #   helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_emb, start_tokens, end_token)
                #   #my_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state, output_layer = output_layer)
                #   my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state, output_layer = output_layer)
                #outputs, c_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(my_decoder,maximum_iterations=10,impute_finished=False,scope=decoder_scope)
                #target_id, sequence_length = None, None
                else:
                    attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
                    alignment_history = False
                    cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = FLAGS.dim_attention, alignment_history=alignment_history, output_attention = True, name='attention')
                    initial_state = cell.zero_state(tf.shape(encoder_sequence_length)[0],tf.float32).clone(cell_state = encoder_state)
                    #cell = decoder_cell
                    #initial_state = encoder_state
                    if FLAGS.predict_with_constrain:
                        my_decoder = ConstrainedDecoder(cell,initial_state,decoder_emb,start_tokens, end_token, constrained_matrix, output_layer = output_layer) 
                    else:
                        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_emb, start_tokens, end_token)
                        my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state, output_layer = output_layer)
                outputs, c_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(my_decoder, maximum_iterations=FLAGS.decoder_max_iter, impute_finished=False, scope=decoder_scope)
                target_id, sequence_length = None, None
                
                if beam_width > 0:
                    #logits = tf.no_op()
                    logits = outputs.scores
                    #logits = c_state.accumulated_attention_probs
                    #logits = outputs.tmp
                    sample_id = outputs.predicted_ids
                    sequence_length = final_sequence_length
                else:
                    logits = outputs.rnn_output
                    sample_id = tf.cast(outputs.sample_id,tf.int64)
                    sequence_length = final_sequence_length
        return logits, sample_id, c_state, target_id, sequence_length

    def build_cell(self, prefix, units):
        with tf.variable_scope('rnn_cell_' + prefix):
            cell = tf.contrib.rnn.GRUCell(units)
        return cell

    def attention_mechanism_fn(self, encoder_outputs, encoder_sequence_length):
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(FLAGS.dim_attention, encoder_outputs, memory_sequence_length = encoder_sequence_length)
        return attention_mechanism

    def calc_loss(self, inference_res):
    #"""Compute optimization loss."""
        logits, target_output, sequence_length,sample_weight = inference_res
        max_time = tf.shape(target_output)[1]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        #crossent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_output, logits = logits)
        #target_weights = tf.sequence_mask(sequence_length, max_time, dtype=logits.dtype)
        #loss = tf.reduce_sum(crossent * target_weights) / tf.cast(tf.shape(target_output)[0],tf.float32)
        target_weights = tf.sequence_mask(sequence_length, max_time, dtype=tf.bool)
        #loss = tf.reduce_sum(tf.where(target_weights, crossent, tf.zeros_like(crossent))) #/ tf.cast(tf.shape(target_output)[0],tf.float32)
        if FLAGS.training_with_weight:
            loss = tf.reduce_sum(tf.where(target_weights, crossent, tf.zeros_like(crossent)),axis=-1)
            loss = tf.reduce_sum(tf.multiply(loss, sample_weight))
            weight = tf.multiply(tf.cast(sequence_length,tf.float32),sample_weight)
            weight = tf.reduce_sum(weight)
        else:
            loss = tf.reduce_sum(tf.where(target_weights, crossent, tf.zeros_like(crossent)))
            weight = tf.cast(tf.reduce_sum(sequence_length),tf.float32)
        
        #return tf.reduce_sum(loss), tf.reduce_sum(tf.cast(tf.reduce_sum(sequence_length),tf.float32))
        return [loss], weight
    
    def get_optimizer(self):
        return [tf.train.GradientDescentOptimizer(FLAGS.learning_rate)]
    def calc_score(self, inference_res):
        logits, target_output, sequence_length, _ = inference_res
        if FLAGS.beam_width:
            #return tf.zeros([tf.shape(sequence_length)[0]])
            #logits: [batch_size, seq_len, beam_size]
            #sequence_length: [batch_size, beam_size]
            #output: [batch_size, beam_size]
            #score = tf.shape(logits)
            #indices: [batch_id, seqLen, beam_id]
            batch_size,maxLen, beam_size = tf.shape(logits)[0],tf.shape(logits)[1],tf.shape(logits)[2]
            batch_id = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size),-1),[1,beam_size]),[-1,1])
            seq_len = tf.reshape(sequence_length-1,[-1,1])
            beam_id = tf.expand_dims(tf.tile(tf.range(beam_size),[batch_size]),-1)
            indices = tf.concat([batch_id,seq_len,beam_id],axis=1)
            score = tf.reshape(tf.gather_nd(logits, indices),[batch_size,-1]) / tf.cast(sequence_length,dtype=tf.float32) 
            #score:[batch_size,beam_width], corresponding to the target_output
            return tf.exp(score)
            #score = tf.reshape(tf.gather_nd(logits,indices),[batch_size,-1])
            #return score
        else:
            max_time = tf.shape(target_output)[1]
            #crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
            softmax_res = tf.log(tf.nn.softmax(logits))
            idx = tf.where(~tf.is_nan(tf.cast(target_output,tf.float32)))
            values = tf.reshape(target_output,[-1,1])
            idx = tf.concat([idx,values],axis=-1)
            prob = tf.reshape(tf.gather_nd(softmax_res, idx),tf.shape(target_output))
            #return prob
            target_weights = tf.sequence_mask(sequence_length, max_time, dtype=tf.bool)
            score = tf.reduce_sum(tf.where(target_weights, prob, tf.zeros_like(prob)),axis=1)/tf.cast(sequence_length,tf.float32)# / tf.cast(tf.reduce_sum(sequence_length),tf.float32)
            return tf.exp(score)#, target_output, softmax_res
    def lookup_infer(self, inference_res):
        sample_id = inference_res[1]
        if not FLAGS.beam_width:
            sample_id_padding = tf.pad(sample_id, [[0,0],[0,FLAGS.decoder_max_iter - tf.shape(sample_id)[1]]])
        else:
            sample_id_padding = tf.pad(sample_id, [[0,0],[0,FLAGS.decoder_max_iter - tf.shape(sample_id)[1]],[0,0]])
        reverse_id = self.reverse_decoder_dict.lookup(tf.to_int64(sample_id_padding))
        return reverse_id,inference_res[2]


if __name__ == '___main__':
    from utils.data_reader import InputPipe
    pred_pipe = InputPipe(FLAGS.input_validation_data_path + "/Normalized_TrainData.tsv", FLAGS.eval_batch_size,1,2,"",True)
    #query,q = pred_pipe.get_next()
    query = tf.placeholder(tf.string)
    decoder_emb = tf.get_variable(name='decoder_emb', shape=[FLAGS.decoder_vocab_size, FLAGS.dim_decoder_emb])
    decoder_dict = lookup_ops.index_table_from_file(FLAGS.output_model_path + "/" + FLAGS.decoder_vocab_file, default_value=UNK_ID)
    target_emb,step_mask,sequence_length,target_id = term_emb_extract(q, decoder_dict, decoder_emb, FLAGS.dim_decoder_emb,add_terminator=True)
    id_shape = tf.shape(target_id)
    emb_shape = tf.shape(target_emb)
    cnt = 0
    with tf.Session() as sess:
        scope = tf.get_variable_scope()
        scope.reuse_variables()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(pred_pipe.iterator.initializer)
        while True:
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)
            query,sh1,sh2 = sess.run([q, id_shape, emb_shape])
            if not sh1[1] == sh2[1]:
                print(query)
        #print(sh1[1])
        #print(sh2[1])

if __name__ == '__main__':
    from utils.data_reader import InputPipe
    #pred_pipe = InputPipe(FLAGS.input_validation_data_path + "/Normalized_TrainData.tsv", FLAGS.eval_batch_size,1,2,"",True)
    pred_pipe = InputPipe(FLAGS.input_validation_data_path, 5, 1, 3, "","","", True)
    #pred_pipe = InputPipe("../Eval_data/test.tsv",1,1,2,"",True)
    batch_input = pred_pipe.get_next()
    model = Seq2Seq()
    inference_res = model.inference(batch_input, tf.contrib.learn.ModeKeys.INFER)
    logits, target_output, sequence_length, c_state = inference_res
    c_state_shape = tf.shape(c_state)
    s = model.calc_score(inference_res)
    #s = tf.shape(model.tmp)
    rewrite,seq_len = model.lookup_infer(inference_res)
    scope = tf.get_variable_scope()
    scope.reuse_variables()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    with tf.Session(config = config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(pred_pipe.iterator.initializer)
        ckpt = tf.train.get_checkpoint_state(FLAGS.input_previous_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Load model from ", ckpt.model_checkpoint_path)
        else:
            print("No initial model found.")
        #print(sess.run([batch_input,logits,target_output,rewrite,seq_len,s]))
        print(sess.run([batch_input,c_state,c_state_shape]))
