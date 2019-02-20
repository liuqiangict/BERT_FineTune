import sys
sys.path.append('./')
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from models.base_model import BaseModelConfig, BaseModel
from module.bert import BertConfig, BertModel, get_assignment_map_from_checkpoint
from module.layers import term_id_extract, mask_maxpool, mask_avgpool
import math

class BertQKConfig(BaseModelConfig):
    def __init__(self,config_dict,config_name='bert_qk'):
        super(BertQKConfig,self).__init__(config_name)
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
        self.dim_last_layer = 0
        self.train_mode = 'neg_sample' #label_finetune
        self.seqemb_mode = 'start_emb' #'maxpool', 'avgpool'
        self.negative_sample = 4
        self.softmax_gamma = 10
        #from main
        self.vocab_path = ""
        #fix param
        self.init_config(config_dict)
        self.print_config()

class BertQK(BaseModel):
    def __init__(self,config_dict):
        super(BertQK,self).__init__()
        self.cfg = BertQKConfig(config_dict)
        #Dictionary initialization
        self.bert_dict = lookup_ops.index_table_from_file(self.cfg.vocab_path + "/" + self.cfg.bert_vocab_file, default_value=0)
    def inference(self, input_fields, mode):
        query = input_fields['query:bertseq'][0]
        query_vec = self.vector_generation(query, mode,'Q')
        doc, doc_vec = None,None
        if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
            doc = input_fields['doc:bertseq'][0]
            doc_vec = self.vector_generation(doc,mode,'D')
        if self.cfg.train_mode == 'label_fintune' and mode == tf.contrib.learn.ModeKeys.TRAIN:
            return query_vec, doc_vec, input_fields['label']
        return query_vec, doc_vec
            
    def vector_generation(self, text, mode, model_prefix):
        ids, mask = term_id_extract(text, self.bert_dict)
        TBert = BertModel(self.cfg, mode == tf.contrib.learn.ModeKeys.TRAIN, input_ids = ids, input_mask = mask, token_type_ids = None, use_one_hot_embeddings = False, scope = 'bert')
        if self.cfg.seqemb_mode == "start_emb":
            vec = TBert.get_pooled_output()
        else:
            vecs = TBert.get_sequence_output()
            if self.cfg.seqemb_mode == "maxpool":
                vec = mask_maxpool(vecs, mask)
            elif self.cfg.seqemb_mode == "avgpool":
                vec = mask_avgpool(vecs, mask)
        if self.cfg.dim_last_layer:
            vec = tf.layers.dense(vec, self.cfg.dim_last_layer, activation = tf.tanh, name=model_prefix + "_dense")
        vec = tf.nn.l2_normalize(vec, dim=1)
        return vec

    def calc_loss(self, inference_res):
        query_vec = inference_res[0]
        doc_vec = inference_res[1]
        label = None if len(inference_res) == 2 else inference_res[2]
        batch_size = tf.shape(query_vec)[0]
        if label == None:
            posCos = tf.reduce_sum(tf.multiply(query_vec, doc_vec), axis = 1)
            allCos = [posCos]
            for i in range(0, self.cfg.negative_sample):
                random_indices = (tf.range(batch_size) + tf.random_uniform([batch_size],1,batch_size,tf.int32)) % batch_size
                negCos = tf.reduce_sum(tf.multiply(query_vec, tf.gather(doc_vec, random_indices)),axis=1)
                allCos.append(tf.where(tf.equal(negCos,1),tf.zeros_like(negCos),negCos))
            allCos = tf.stack(allCos, axis=1)
            softmax = tf.nn.softmax(allCos * self.cfg.softmax_gamma, dim = 1)
            loss = tf.reduce_sum(-tf.log(softmax[:,0]))
            weight = batch_size
            tf.summary.scalar('softmax_losses',loss)
        else:
            pred = tf.reduce_sum(tf.multiply(query_vec, doc_vec), axis=1)
            pred = tf.nn.relu(pred)
            loss = tf.losses.log_loss(labels = tf.string_to_number(label), predictions=pred, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
            weight = batch_size
            tf.summary.scalar('log_losses',loss)
        return [loss], weight
    def lookup_infer(self, inference_res):
        query_vec, doc_vec = inference_res
        batch_size = tf.shape(query_vec)[0]
        vec_len = tf.shape(query_vec)[1]
        return query_vec, tf.ones([batch_size],dtype = tf.int32) * vec_len
    def calc_score(self, inference_res):
        query_vec, doc_vec = inference_res
        #if FLAGS.mode == "predict":
        #    batch_size = tf.shape(query_vec)[0]
        #    return tf.ones([batch_size])
        if doc_vec == None:
            batch_size = tf.shape(query_vec)[0]
            return tf.ones([batch_size])
        score = tf.reduce_sum(tf.multiply(query_vec, doc_vec), axis = 1)
        #score = tf.norm(query_vec-doc_vec,axis=1)
        return score


if __name__ == '__main__':
    from utils.data_reader import InputPipe
    pred_pipe = InputPipe(FLAGS.input_validation_data_path, 10, 1, 3, "","0,1","",True)
    batch_input = pred_pipe.get_next()
    model = BertQKModel()
    inference_res = model.inference(batch_input, tf.contrib.learn.ModeKeys.EVAL)
    
    #logits, target_output, sequence_length, _ = inference_res
    s = model.calc_score(inference_res)
    #s = tf.shape(model.tmp)
    #rewrite,seq_len = model.lookup_infer(inference_res)
    scope = tf.get_variable_scope()
    scope.reuse_variables()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    with tf.Session(config = config) as sess:
        #sess.run(tf.local_variables_initializer())
        #sess.run(tf.tables_initializer())
        #sess.run(tf.global_variables_initializer())
        #sess.run(pred_pipe.iterator.initializer)
        ckpt = tf.train.get_checkpoint_state(FLAGS.input_previous_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            tvars = tf.trainable_variables()
            #print(tvars)
            #(assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, "../initial_models/initial_model_bert/bert_model.ckpt")
            tf.train.init_from_checkpoint(ckpt.model_checkpoint_path,{"bert/":"bert/"})
            #tf.train.init_from_checkpoint(ckpt.model_checkpoint_path,assignment_map)
            #    assignment_map[k] = assignment_map[k] + ":0"
            #print(assignment_map)
            #tf.train.init_from_checkpoint("../initial_models/initial_model_bert/bert_model.ckpt", assignment_map)
            #saver.restore(sess, ckpt.model_checkpoint_path)
            print("Load model from ", ckpt.model_checkpoint_path)
        #sess.run(tf.get_all_variables())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(pred_pipe.iterator.initializer)

        print(sess.run([batch_input,s]))
