import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core
import sys
sys.path.append('.')
from models.base_model import BaseModelConfig, BaseModel
from module.constrained_decoder import ConstrainedDecoder, generate_constrained_matrix
from module.constrained_beam_search_decoder import ConstrainedBeamSearchDecoder, tile_batch
from module.layers import term_id_extract, term_emb_extract
from module.bert import BertModel
from utils.fix_param import DictParam
#dict config has high priority
class Bert2SeqConfig(BaseModelConfig):
    def __init__(self,config_dict,config_name='bert2seq'):
        super(Bert2SeqConfig,self).__init__(config_name)
        #from json
        self.attention_probs_dropout_prob = 0.1
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.initializer_range = 0.02
        self.intermediate_size = 3072
        self.max_position_embeddings = 512
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.type_vocab_size = 2
        self.vocab_size = 30522
        self.bert_vocab_file = 'vocab.txt'
        self.decoder_vocab_file = ""
        self.decoder_vocab_size = 200003
        self.training_with_weight = 0
        self.predict_with_constrain = 0
        self.dim_attention = 768
        self.dim_decoder_emb = 64
        self.beam_width = 1
        self.length_penalty_weight = 0
        self.coverage_penalty_weight = 0
        self.decoder_max_iter = 10
        #from main
        self.vocab_path = ""
        #fix param
        self.dict_param = DictParam()
        self.init_config(config_dict)
        self.print_config()

class Bert2Seq(BaseModel):
    def __init__(self, config_dict):
        super(Bert2Seq,self).__init__()
        self.cfg = Bert2SeqConfig(config_dict)
        #Dictionary initialization
        self.bert_dict = lookup_ops.index_table_from_file(self.cfg.vocab_path + "/" + self.cfg.bert_vocab_file, default_value=0)
        self.decoder_dict = lookup_ops.index_table_from_file(self.cfg.vocab_path + "/" + self.cfg.decoder_vocab_file, default_value=self.cfg.dict_param.unk_id)
        self.reverse_decoder_dict = lookup_ops.index_to_string_table_from_file(self.cfg.vocab_path + "/" + self.cfg.decoder_vocab_file, default_value=self.cfg.dict_param.unk)
    def inference(self, input_fields, mode):
        if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
            query, doc = input_fields["query:bertseq"][0], input_fields["doc"]
        else:
            query,doc = input_fields["query:bertseq"][0], None
        if self.cfg.training_with_weight and mode == tf.contrib.learn.ModeKeys.TRAIN:
            weight = tf.string_to_number(input_fields["weight"])
        else:
            weight = None
        if self.cfg.predict_with_constrain and mode == tf.contrib.learn.ModeKeys.INFER:
            doc = input_fields["paragragh"]
        output_state, c_state, source_sequence_length = self.encoder(query,mode)
        logits, sample_id, final_state, target_id, target_sequence_length = self.decoder(output_state, c_state, source_sequence_length, doc, mode)
        if not mode == tf.contrib.learn.ModeKeys.INFER:
            return [logits, target_id, target_sequence_length, weight]
        else:
            return [logits, sample_id, target_sequence_length, weight]
    
    def encoder(self, query, mode):
        q_ids,q_mask = term_id_extract(query, self.bert_dict)
        q_len = tf.count_nonzero(q_mask,axis=-1)
        QBert = BertModel(self.cfg, mode == tf.contrib.learn.ModeKeys.TRAIN, input_ids=q_ids, input_mask=q_mask, token_type_ids=None,use_one_hot_embeddings=False,scope="bert")
        final_output = QBert.get_pooled_output() #hidden layer is ...12 layers =(
        output = QBert.get_sequence_output()
        if not self.cfg.dim_attention == self.cfg.hidden_size:
            output_small = tf.layers.dense(final_output,self.cfg.dim_attention,activation=tf.nn.relu,reuse=tf.AUTO_REUSE)
            final_output = (final_output,output_small)
        return output, final_output, q_len 

    def decoder(self, encoder_outputs, encoder_state, encoder_sequence_length, doc, mode):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE) as decoder_scope:
            decoder_emb = tf.get_variable(name='decoder_emb', shape=[self.cfg.decoder_vocab_size, self.cfg.dim_decoder_emb])
            decoder_cell = self.build_cell('decoder_cell', self.cfg.hidden_size)
            if not self.cfg.dim_attention == self.cfg.hidden_size:
                decoder_cells = [decoder_cell,self.build_cell('decoder_1', self.cfg.dim_attention)]
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cells,state_is_tuple=True)
            output_layer = layers_core.Dense(self.cfg.decoder_vocab_size, name='output_projection',kernel_initializer=self.default_init())
            if not mode == tf.contrib.learn.ModeKeys.INFER: #train and eval
                target_input = doc
                attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
                cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = self.cfg.dim_attention, alignment_history=False, output_attention = True, name='attention')
                initial_state = cell.zero_state(tf.shape(encoder_sequence_length)[0],tf.float32).clone(cell_state = encoder_state)
                target_emb,step_mask,sequence_length,target_id = term_emb_extract(target_input, self.decoder_dict, decoder_emb, self.cfg.dim_decoder_emb,add_terminator=True)
                sos_emb = tf.tile(tf.nn.embedding_lookup(decoder_emb,self.cfg.dict_param.sos_id),tf.stack([tf.shape(sequence_length)[0]]))
                sos_emb = tf.reshape(sos_emb,tf.stack([tf.shape(sequence_length)[0],1,self.cfg.dim_decoder_emb]))
                target_emb = tf.concat([sos_emb,target_emb],axis=1)
                sequence_length = sequence_length + 1
                helper = tf.contrib.seq2seq.TrainingHelper(target_emb,sequence_length,time_major=False)
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state)
                outputs, c_state, _ = tf.contrib.seq2seq.dynamic_decode(my_decoder, output_time_major = False,impute_finished = False,scope=decoder_scope)
               #??
                sample_id = outputs.sample_id
                logits = output_layer(outputs.rnn_output)
            else:
                beam_width = self.cfg.beam_width
                start_tokens = tf.fill([tf.shape(encoder_outputs)[0]],self.cfg.dict_param.sos_id)
                end_token = self.cfg.dict_param.eos_id
                if self.cfg.predict_with_constrain:
                    generate_constrain_matrix(doc,self.decoder_dict,self.cfg.decoder_vocab_size)
                else:
                    constrained_matrix = None
                if beam_width > 1:
                    #encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
                    #encoder_sequence_length = tf.contrib.seq2seq.tile_batch(encoder_sequence_length, multiplier=beam_width)
                    #encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
                    true_batch_size = tf.shape(encoder_sequence_length)[0]
                    encoder_outputs = tile_batch(encoder_outputs, multiplier=beam_width)
                    encoder_sequence_length = tile_batch(encoder_sequence_length, multiplier=beam_width)
                    encoder_state = tile_batch(encoder_state, multiplier=beam_width)
                    attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
                    cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = self.cfg.dim_attention, alignment_history=False, output_attention = True, name='attention')
                    initial_state = cell.zero_state(batch_size=true_batch_size * beam_width, dtype=tf.float32)
                    initial_state = initial_state.clone(cell_state=encoder_state)
                    my_decoder = ConstrainedBeamSearchDecoder(
                            cell = cell,
                            embedding=decoder_emb,
                            start_tokens = start_tokens,
                            end_token = end_token,
                            initial_state=initial_state,
                            beam_width=beam_width,
                            constrained_matrix = constrained_matrix,
                            output_layer = output_layer,
                            length_penalty_weight=self.cfg.length_penalty_weight,
                            coverage_penalty_weight = self.cfg.coverage_penalty_weight,
                            reorder_tensor_arrays = True
                        )
                else:
                    attention_mechanism = self.attention_mechanism_fn(encoder_outputs, tf.maximum(encoder_sequence_length,tf.ones_like(encoder_sequence_length)))
                    alignment_history = False
                    cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size = self.cfg.dim_attention, alignment_history=alignment_history, output_attention = True, name='attention')
                    initial_state = cell.zero_state(tf.shape(encoder_sequence_length)[0],tf.float32).clone(cell_state = encoder_state)
                    if self.cfg.predict_with_constrain:
                        my_decoder = ConstrainedDecoder(cell,initial_state,decoder_emb,start_tokens, end_token, constrained_matrix, output_layer = output_layer) 
                    else:
                        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_emb, start_tokens, end_token)
                        my_decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, initial_state, output_layer = output_layer)
                outputs, c_state, final_sequence_length = tf.contrib.seq2seq.dynamic_decode(my_decoder, maximum_iterations=self.cfg.decoder_max_iter, impute_finished=False, scope=decoder_scope)
                target_id, sequence_length = None, None
                
                if beam_width > 1:
                    logits = outputs.scores
                    sample_id = outputs.predicted_ids
                    sequence_length = c_state.lengths#This is important since dynamic_docode could not track length right. #final_sequence_length
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
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.cfg.dim_attention, encoder_outputs, memory_sequence_length = encoder_sequence_length)
        return attention_mechanism

    def calc_loss(self, inference_res):
    #"""Compute optimization loss."""
        logits, target_output, sequence_length,sample_weight = inference_res
        max_time = tf.shape(target_output)[1]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(sequence_length, max_time, dtype=tf.bool)
        if self.cfg.training_with_weight:
            loss = tf.reduce_sum(tf.where(target_weights, crossent, tf.zeros_like(crossent)),axis=-1)
            loss = tf.reduce_sum(tf.multiply(loss, sample_weight))
            weight = tf.multiply(tf.cast(sequence_length,tf.float32),sample_weight)
            weight = tf.reduce_sum(weight)
        else:
            loss = tf.reduce_sum(tf.where(target_weights, crossent, tf.zeros_like(crossent)))
            weight = tf.cast(tf.reduce_sum(sequence_length),tf.float32)
        return [loss], weight

    def calc_score(self, inference_res):
        logits, target_output, sequence_length, _ = inference_res
        if self.cfg.beam_width < 0 : #wrong beam search score version
            #return tf.zeros([tf.shape(sequence_length)[0]])
            #logits: [batch_size, seq_len, beam_size]
            #sequence_length: [batch_size, beam_size]
            #output: [batch_size, beam_size]
            #score = tf.shape(logits)
            #indices: [batch_id, seqLen, beam_id]

            batch_size,maxLen, beam_size = tf.shape(logits)[0],tf.shape(logits)[1],tf.shape(logits)[2]
            batch_id = tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size),-1),[1,beam_size]),[-1,1])
            seq_len = tf.cast(tf.reshape(sequence_length-1,[-1,1]),tf.int32)
            beam_id = tf.expand_dims(tf.tile(tf.range(beam_size),[batch_size]),-1)
            indices = tf.concat([batch_id,seq_len,beam_id],axis=1)
            score = tf.reshape(tf.gather_nd(logits, indices),[batch_size,-1]) / tf.cast(sequence_length,dtype=tf.float32) 
            #score:[batch_size,beam_width], corresponding to the target_output
            return tf.exp(score)
            #score = tf.reshape(tf.gather_nd(logits,indices),[batch_size,-1])
            #return score
        elif self.cfg.beam_width > 1:
            score = tf.div(logits[:,-1,:],tf.cast(sequence_length,dtype=tf.float32)) 
            return tf.exp(score)
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
        if self.cfg.beam_width <= 1:
            sample_id_padding = tf.pad(sample_id, [[0,0],[0,self.cfg.decoder_max_iter - tf.shape(sample_id)[1]]])
        else:
            sample_id_padding = tf.pad(sample_id, [[0,0],[0,self.cfg.decoder_max_iter - tf.shape(sample_id)[1]],[0,0]])
        reverse_id = self.reverse_decoder_dict.lookup(tf.to_int64(sample_id_padding))
        return reverse_id,inference_res[2]

    def get_optimizer(self, opt_type, lr):
        return [self.choose_optimizer(opt_type, lr)]
