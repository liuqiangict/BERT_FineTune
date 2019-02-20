import tensorflow as tf
import sys
sys.path.append('.')
from utils.preprocessor import XletterPreprocessor
from utils.preprocessor import BertPreprocessor
from utils.helper import BaseConfig
#mode: train, eval_auc, eval_bleu, infer, score

class InputConfig(BaseConfig):
    def __init__(self, cfg, input_mode, config_name):
        super(InputConfig,self).__init__(config_name)
        #self.cfg = cfg
        #fix params
        self.all_preprocessor = set(['xletter','bertseq','bertpair'])
        self.pair_preprocessor = set(['bertpair'])
        #init from cfg
        self.input_mode = input_mode
        self.batch_size = cfg[input_mode+"_batch_size"]
        self.num_epochs = cfg["num_epochs"] if input_mode == 'train' else 1
        self.header = cfg[input_mode+"_header"].split(',')
        self.append_ori = False if self.input_mode == 'train' or self.input_mode.startswith('eval') else True
        self.read_thread = cfg.get('read_thread',20)
        self.shuffle_buffer_size = cfg.get('shuffle_buffer_size',10)
        self.prefetch_buffer_size = cfg.get('prefetch_buffer_size',8)
        #generated params
        self.preprocess_header = []
        self.preprocess_helper = []
        self.preprocessor_name = set()
        self.outheader = self.header.copy()
        folder = cfg["input_training_data_path"] if input_mode == 'train' else cfg["input_validation_data_path"]
        self.filenames = self.get_filenames(folder + cfg[input_mode+"_filename"])
        #init preprocessor config
        self.parse_preprocess_fields()
        if "bertseq" in self.preprocessor_name or "bertpair":
            self.vocab_path = cfg.get('vocab_path', None)
            self.bert_vocab_file = cfg.get('bert_vocab_file', None)
            self.bert_lower_case = cfg.get('bert_lower_case',None)
            self.max_position_embeddings = cfg.get('max_position_embeddings',None)
        if "xletter" in self.preprocessor_name:
            self.xletter_dict = cfg['xletter_dict'] if 'xletter_dict' in cfg else None
            self.xletter_win_size = cfg['xletter_win_size'] if 'xletter_win_size' in cfg else None
        #print config
        self.notprint |= set(["all_preprocessor","pair_preprocessor"])
        self.print_config()
    def get_filenames(self,raw_name):
        if tf.gfile.IsDirectory(raw_name):
            return [raw_name + "/" + i for i in tf.gfile.ListDirectory(raw_name)]
        else:
            return raw_name
    def parse_preprocess_fields(self):
        record = dict()
        for idx,col in enumerate(self.header):
            self.preprocess_header.append(None)
            self.preprocess_helper.append(None)
            f = col.split(':')
            if len(f) > 1:
                if f[1] not in self.all_preprocessor:
                    raise NotImplementedError
                self.preprocessor_name.add(f[1])
                if self.append_ori:
                    self.outheader.append(f[0])
                if len(f) == 2 or int(f[3]) == 0: #0:colname, 1:preprocessor_name, 2:preprocessor_id, 3: column_id
                    if not len(f) == 2:
                        record[f[1]+"_"+f[2]] = idx
                    self.preprocess_header[-1] = f[1]
                else: # column_id not 0
                    self.preprocess_header[-1] = "" #return none
                    self.preprocess_helper[record[f[1]+"_"+f[2]]] = idx

class InputPipe():
    def __init__(self, input_mode, cfg):
        self.cfg = InputConfig(cfg, input_mode, input_mode+"_input_pipe") 
        if len(self.cfg.preprocessor_name): #init preprocessor
            self.preprocessor = dict()
            for p in self.cfg.preprocessor_name:
                self.preprocessor[p] = self.init_preprocess_func(p)

        ds = tf.data.TextLineDataset(self.cfg.filenames)
        if self.cfg.input_mode == 'train':
            ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.cfg.shuffle_buffer_size*self.cfg.batch_size, count = self.cfg.num_epochs))
        ds_batch = ds.apply(tf.data.experimental.map_and_batch(map_func=self.parse_line, batch_size=self.cfg.batch_size, drop_remainder=False, num_parallel_calls=self.cfg.read_thread))
        ds_batch = ds_batch.prefetch(buffer_size=self.cfg.prefetch_buffer_size)
        self.iterator = ds_batch.make_initializable_iterator()

    def parse_line(self, line):
        columns = tf.decode_csv(line, [[""] for h in self.cfg.header],field_delim="\t",use_quote_delim=False)
        if len(self.cfg.preprocessor_name):
            columns = self.preprocess(columns)
        return dict(zip(self.cfg.outheader, columns))

    def init_preprocess_func(self,func_name):
        if func_name=='xletter':
            self.xletter_preprocessor = XletterPreprocessor(self.cfg.xletter_dict, self.cfg.xletter_win_size)
            return lambda x: tuple(tf.py_func(self.xletter_preprocessor.xletter_extractor,[x],[tf.string]))
        if func_name=='bertseq':
            self.bertseq_preprocessor = BertPreprocessor(self.cfg.vocab_path + '/' + self.cfg.bert_vocab_file, self.cfg.bert_lower_case,self.cfg.max_position_embeddings)
            return lambda x: tuple(tf.py_func(self.bertseq_preprocessor.single_seq_extractor,[x],[tf.string]))
        if func_name=='bertpair':
            self.bertpair_preprocessor = BertPreprocessor(self.cfg.vocab_path + '/' + self.cfg.bert_vocab_file, self.cfg.bert_lower_case,self.cfg.max_position_embeddings)
            return lambda x,y: tuple(tf.py_func(self.bertpair_preprocessor.seq_pair_extractor,[x,y],[tf.string,tf.string])) 
        return NotImplementedError

    def get_header(self):
        return self.cfg.outheader
    
    def preprocess(self, column):
        res = []
        ori = []
        for i,p in enumerate(self.cfg.preprocess_header):
            if p == None:
                res.append(column[i])
            elif not len(p):
                res.append(column[i])
            elif p in self.cfg.pair_preprocessor:
                res.append(self.preprocessor[p](column[i],column[self.cfg.preprocess_helper[i]]))
            else:
                res.append(self.preprocessor[p](column[i]))
            if not p == None and self.cfg.append_ori:
                ori.append(column[i])
        return res + ori


    #def preprocess(self, column):
    #    res = []
    #    for i,col in enumerate(column):
    #        if i in self.xf:
    #            res.append(tuple(tf.py_func(self.xletter_preprocessor.xletter_extractor,[column[i]],[tf.string])))
    #        elif i in self.bf:
    #            res.append(tuple(tf.py_func(self.bert_preprocessor.single_seq_extractor,[column[i]],[tf.string])))
    #        elif i in self.bspf:
    #            res.append(tuple(tf.py_func(self.bert_preprocessor.seq_pair_extractor,[column[i]],[tf.string])))
    #        else:
    #            res.append(column[i])
    #    if self.append_ori:
    #        res += [column[i] for i in self.xf + self.bf + self.bspf]
    #    return res
    def get_next(self):
        return self.iterator.get_next()

if __name__ == '__main__':
    from config.param import FLAGS
    from main import move_file
    from main import get_config
    cfg = get_config()
    cfg["infer_header"] = "query:bertpair:0:0,doc:bertpair:0:1"
    cfg["infer_batch_size"] = 5
    #print("all config:",cfg)
    move_file(cfg)
    inp = InputPipe('infer',cfg)
    with tf.Session() as sess:
        sess.run(inp.iterator.initializer)
        for i in range(0,3):
            print(i)
            print(sess.run(inp.get_next())) 
