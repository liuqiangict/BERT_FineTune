import json
import tensorflow as tf
import sys
sys.path.append('.')
from utils.helper import BaseConfig
import tensorflow.contrib.microsoft as mstf
"""Base class for models"""

class BaseModelConfig(BaseConfig):
    def __init__(self, config_name, config_dict=None):
        super(BaseModelConfig,self).__init__(config_name)
class BaseModel:
    def __init__(self, dict_config=None):
        self.loss_cnt = 1
    def inference(self, input_fileds, mode):
        raise NotImplementedError
    def calc_loss(self, inference_res):
        raise NotImplementedError
    def calc_score(self, inference_res):
        raise NotImplementedError
    def lookup_infer(self, inference_res):
        raise NotImplementedError
    #Common Function
    def get_optimizer(self,opt_type,lr):
        return [self.choose_optimizer(opt_type,lr)]
    def choose_optimizer(self,opt_type,lr):
        if opt_type == 'sgd':
            opt = tf.train.GradientDescentOptimizer(lr)
        elif opt_type == 'adam':
            opt = tf.train.AdamOptimizer(lr)
            #opt =  tf.contrib.optimizer_v2.AdamOptimizer(lr)
        else:
            NotImplementedError
        return opt
    def get_ophelper(self, input_mode, xletter_dict = None, xletter_win_size = 3 ):
        if input_mode == 'mstf':
            return mstf.dssm_dict(xletter_dict)
        if input_mode == 'pyfunc_batch':
            return XletterPreprocessor(xletter_dict, xletter_win_size)
        return None
    def default_init(self):
        return tf.contrib.layers.variance_scaling_initializer(factor=1.0,mode="FAN_AVG",uniform=True)
