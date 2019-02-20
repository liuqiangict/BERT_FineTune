import tensorflow as tf
#Interface
tf.app.flags.DEFINE_string('input-training-data-path','../Data/Train/QK/','training data path')
#tf.app.flags.DEFINE_string('input-training-data-path','../../S2S/Train_Data','training data path')
#tf.app.flags.DEFINE_string('input-training-data-path','../TreeRetrieve_data\QK', 'training data path')
tf.app.flags.DEFINE_string('input-validation-data-path','../Data/Eval/label_data.txt', 'validation path')
#tf.app.flags.DEFINE_string('input-validation-data-path','../Eval_Data/test.txt', 'validation path')
#tf.app.flags.DEFINE_string('input-validation-data-path','../Eval_Data/Normalized_TrainData.tsv','validation path')
#tf.app.flags.DEFINE_string('input-validation-data-path','../Eval_Data/CO_Test.tsv','validation path')
#tf.app.flags.DEFINE_string('input-validation-data-path','../Eval_Data/Transformer_candidates_v2.tsv', 'validation path')
#tf.app.flags.DEFINE_string('input-previous-model-path','../initial_models/initial_model_esl','initial model path')
tf.app.flags.DEFINE_string('input-previous-model-path','../Model/initial_model','initial model path')
#tf.app.flags.DEFINE_string('input-previous-model-path','finalmodel','initial model path')
tf.app.flags.DEFINE_string('output-model-path','../Model/finalmodel','path to save model')
#tf.app.flags.DEFINE_string('output-model-path','../initial_models/initial_model_bert','path to save model')
tf.app.flags.DEFINE_string('log-dir','../Log/log_folder','folder to save log')

#msft cdssm operator
tf.app.flags.DEFINE_integer('use-mstf-ops',1, 'whether to use mstf operator: 1: use, 0: not use and preprocess xletter in reader, -1: faster than 0')
tf.app.flags.DEFINE_string('xletter-dict','utils/l3g.txt','xletter dictionary name')
tf.app.flags.DEFINE_integer('xletter-win-size',3,'xletter conv win size')
tf.app.flags.DEFINE_integer('xletter-cnt',49292,'xletter feature num')


#Data Reader Setting
tf.app.flags.DEFINE_integer('read-thread',10,'threads count to read data')
tf.app.flags.DEFINE_integer('buffer-size',10000,'buffer size for data reader')

#Trainer
tf.app.flags.DEFINE_string('mode','train','train, predict or evaluation mode')
tf.app.flags.DEFINE_integer('early-stop-steps',30, 'bad checks to trigger early stop, -1 is to disable early stop')
tf.app.flags.DEFINE_integer('batch-size', 128,'training batch size')
tf.app.flags.DEFINE_integer('eval-batch-size',64,'evaluation batch size')
tf.app.flags.DEFINE_integer('num-epochs',5, 'training epochs')
tf.app.flags.DEFINE_integer('max-model-to-keep',10, 'max models to save')
tf.app.flags.DEFINE_integer('log-frequency', 1000, 'log frequency during training procedure')
tf.app.flags.DEFINE_integer('checkpoint-frequency', 100000, 'evaluation frequency during training procedure')
tf.app.flags.DEFINE_float('learning-rate',0.001, 'learning rate')
tf.app.flags.DEFINE_bool('auc-evaluation',False,'whether to do auc evaluation')
tf.app.flags.DEFINE_bool('bleu-evaluation', False, 'whether to do bleu evaluation')
tf.app.flags.DEFINE_integer('negative-sample',4,'negative sample count')
tf.app.flags.DEFINE_string('metrics-early-stop','auc','metrics to control early stop')
tf.app.flags.DEFINE_integer('loss-cnt',1,'total loss count to update')

#Test settings
tf.app.flags.DEFINE_integer('test-fields',2,'test fields count')
tf.app.flags.DEFINE_string('result-file-name','predict.txt','result file name')

#CDSSM Model
tf.app.flags.DEFINE_string('semantic-model-dims','64', 'semantic model dims, split by ,')
tf.app.flags.DEFINE_integer('dim-xletter-emb', 288, 'xletter embedding dimension')
tf.app.flags.DEFINE_float('softmax-gamma',10.0,'softmax parameters')

#Seq2Seq Model: From xletter to term
tf.app.flags.DEFINE_bool('training-with-weight', 0, 'whether training sample has weight')
tf.app.flags.DEFINE_string('decoder-vocab-file','Term100000.tsv','term vocabulary file')
tf.app.flags.DEFINE_integer('beam-width',0, 'beam search width')
tf.app.flags.DEFINE_float('length-penalty-weight', 0.0, 'length penalty weight')
tf.app.flags.DEFINE_float('coverage-penalty-weight', 0.0, 'coverage penalty weight')
tf.app.flags.DEFINE_integer('dim-decoder', 768, 'decoder dimension')
tf.app.flags.DEFINE_integer('dim-decoder-emb', 128, 'decoder embedding dimension')
tf.app.flags.DEFINE_integer('decoder-vocab-size', 100003, 'decoder vocab size')
tf.app.flags.DEFINE_integer('dim-encoder', 768, 'encoder dimension')
tf.app.flags.DEFINE_integer('dim-attention', 768, 'attention dim')
tf.app.flags.DEFINE_bool('predict-with-constrain', 0 , 'whether predict with constrained')
tf.app.flags.DEFINE_integer('decoder-max-iter', 10, 'max interation limit when decoding')
tf.app.flags.DEFINE_integer('num-hidden-layers', 1, 'hidden layer nums for both encoder and decoder')
#Tree based retrieval
tf.app.flags.DEFINE_bool('leaf-content-emb',True, 'Whether to use leaf content as embedding material')
tf.app.flags.DEFINE_integer('top-k',10,'top K result to return')
tf.app.flags.DEFINE_string('tree-index-file', 'Keyword10k.tsv','tree index file name')
tf.app.flags.DEFINE_integer('layer-negative-sample', 4, 'layer neg sample count')
tf.app.flags.DEFINE_float('layer-weight',0.01,'layer weight for loss')


#Bert
tf.app.flags.DEFINE_bool('bert-lower-case',True, 'True if not case sensitive in bert preprocessor')
tf.app.flags.DEFINE_string('bert-vocab-file','vocab.txt','bert vocab name')
tf.app.flags.DEFINE_integer('bert-vocab-size',30522,'bert vocab size')
tf.app.flags.DEFINE_integer('bert-hidden-size',768,'hidden size')
tf.app.flags.DEFINE_integer('bert-num-hidden-layers',12,'bert hidden layers')
tf.app.flags.DEFINE_integer('bert-num-attention-heads',12,'bert attention heads count')
tf.app.flags.DEFINE_integer('bert-intermediate-size',3072,'bert intermediate size')
tf.app.flags.DEFINE_string('bert-hidden-act','gelu','activate function')
tf.app.flags.DEFINE_float('bert-hidden-dropout-prob',0.1,'bert hidden layers dropout rate')
tf.app.flags.DEFINE_float('bert-attention-probs-dropout-prob',0.1,'bert attention drop out prob')
tf.app.flags.DEFINE_integer('bert-max-position-embeddings',512,'bert intermediate size')
tf.app.flags.DEFINE_integer('bert-type-vocab-size',2,'bert type vocab size')
tf.app.flags.DEFINE_float('bert-initializer-range',0.02,'bert initializer range')
tf.app.flags.DEFINE_integer('bert-max-seq-len',512,'bert max sequence length')
tf.app.flags.DEFINE_integer('bert-seqemb-func',1,'1:first token, 2:max pool, 3:avg pool')
tf.app.flags.DEFINE_integer('bert-qk-output',64,'bert qk model output dim')
tf.app.flags.DEFINE_bool('bert-label-finetune',False,'use label data to finetune')
FLAGS = tf.app.flags.FLAGS
