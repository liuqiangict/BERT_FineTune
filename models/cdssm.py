import tensorflow as tf
import tensorflow.contrib.microsoft as mstf
import math
import sys
sys.path.append('.')
from utils.preprocessor import XletterPreprocessor
from utils.helper import parse_dims
from models.base_model import BaseModelConfig, BaseModel
from module.layers import xletter_feature_extractor, mask_maxpool,mstf_xletter_maxpool
class CDSSMConfig(BaseModelConfig):
    def __init__(self, config_dict, config_name='CDSSM'):
        super(CDSSMConfig,self).__init__(config_name)
        self.input_mode = 'mstf' #'pyfunc, pyfunc_batch'
        self.maxpool_mode = 'mstf' #'emb'
        self.xletter_dict = './utils/l3g.txt'
        self.xletter_win_size = 3
        self.xletter_cnt = 49292
        self.dim_xletter_emb = 288
        self.dim_hidden_layers = "64"
        self.negative_sample = 4
        self.softmax_gamma = 10
        #init config and print
        #params
        self.init_config(config_dict)
        self.print_config()

class CDSSM(BaseModel):
    def __init__(self, config_dict):
        super(CDSSM,self).__init__()
        self.cfg = CDSSMConfig(config_dict)
        self.op_helper = self.get_ophelper(self.cfg.input_mode, self.cfg.xletter_dict, self.cfg.xletter_win_size)
    def inference(self, input_fields, mode):
        query = input_fields['query:xletter'][0] if self.cfg.input_mode == 'pyfunc' else input_fields['query']
        query_vec = self.vector_generation(query, 'Q')
        doc, doc_vec = None,None
        if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
            doc = input_fields['doc:xletter'][0] if self.cfg.input_mode == 'pyfunc' else input_fields['doc']
            doc_vec = self.vector_generation(doc,'D')
        return query_vec, doc_vec
    def vector_generation(self, text, model_prefix):
        dims = parse_dims(self.cfg.dim_hidden_layers)
        if self.cfg.maxpool_mode == 'mstf':
            maxpooling_vec = mstf_xletter_maxpool(text, model_prefix, self.op_helper, self.cfg.xletter_cnt, self.cfg.xletter_win_size, self.cfg.dim_xletter_emb)
        else:
            text_vecs, step_mask, sequence_length = xletter_feature_extractor(text, model_prefix, self.cfg.input_mode,self.op_helper, self.cfg.xletter_cnt, self.cfg.xletter_win_size, self.cfg.dim_xletter_emb)
            maxpooling_vec = mask_maxpool(text_vecs,step_mask)
        dim_input = self.cfg.dim_xletter_emb
        input_vec = tf.nn.tanh(maxpooling_vec)
        for i, dim in enumerate(dims):
            dim_output = dim
            random_range = math.sqrt(6.0/(dim_input+dim_output))
            with tf.variable_scope("semantic_layer{:}".format(i),reuse=tf.AUTO_REUSE):
                weight = tf.get_variable("weight_" + model_prefix, shape = [dim_input, dim_output], initializer = tf.random_uniform_initializer(-random_range, random_range))
                output_vec = tf.matmul(input_vec, weight)
                output_vec = tf.nn.tanh(output_vec)
                input_vec = output_vec
        normalized_vec = tf.nn.l2_normalize(output_vec, dim = 1, name='text_vec')
        return normalized_vec
    
    def calc_loss(self, inference_res):
        query_vec, doc_vec = inference_res
        batch_size = tf.shape(query_vec)[0]
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
        return [loss], weight
    
    def calc_score(self, inference_res):
        query_vec, doc_vec = inference_res
        score = tf.reduce_sum(tf.multiply(query_vec, doc_vec), axis = 1)
        return score
