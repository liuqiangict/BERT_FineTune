import tensorflow as tf
#Interface
tf.app.flags.DEFINE_string('input_training_data_path','../Data/Train/QQ','training data path')
tf.app.flags.DEFINE_string('input_validation_data_path','../Data/Eval/', 'validation path')
tf.app.flags.DEFINE_string('input_previous_model_path','../Model/init_xletter2seq','initial model path')
tf.app.flags.DEFINE_string('output_model_path','../Model/final_model','path to save model')
tf.app.flags.DEFINE_string('log_dir','../Log/log_folder','folder to save log')

tf.app.flags.DEFINE_integer('local',0,'local run, 0:AEther, N: NGPU')
#Model settings
tf.app.flags.DEFINE_string('mode','train','train, predict or evaluation mode')
tf.app.flags.DEFINE_string('modeltype','bert2seq','select model type')
tf.app.flags.DEFINE_bool('auc_eval', False, 'whether to eval AUC')
tf.app.flags.DEFINE_bool('bleu_eval', False, 'whether to eval bleu score')

#Input pipe settings
#Train/Eval
tf.app.flags.DEFINE_integer('num_epochs', 5, 'training epochs')
tf.app.flags.DEFINE_integer('train_batch_size', 32, 'training batch size')
tf.app.flags.DEFINE_string('train_filename', '', 'train file name')
tf.app.flags.DEFINE_string('train_header','query,doc','train header')

tf.app.flags.DEFINE_integer('eval_auc_batch_size', 16, 'auc eval batch size')
tf.app.flags.DEFINE_string('eval_auc_filename', '/label_data.txt', 'auc eval filename')
tf.app.flags.DEFINE_string('eval_auc_header','query,doc,label','auc eval header')

tf.app.flags.DEFINE_integer('eval_bleu_batch_size', 16, 'bleu eval batch size')
tf.app.flags.DEFINE_string('eval_bleu_filename', '/bleu_data.txt', 'bleu eval filename')
tf.app.flags.DEFINE_string('eval_bleu_header','query,doc','bleu eval header')

#Infer/Score
tf.app.flags.DEFINE_integer('infer_batch_size', 16, 'inference batch size')
tf.app.flags.DEFINE_string('infer_filename', '', 'inference file name')
tf.app.flags.DEFINE_string('infer_header','query:bertseq,doc','inference header')

tf.app.flags.DEFINE_integer('score_batch_size', 16, 'scoring batch size')
tf.app.flags.DEFINE_string('score_filename', '', 'score file name')
tf.app.flags.DEFINE_string('score_header','','inference header')

#Data Reader Setting
tf.app.flags.DEFINE_integer('read_thread',30,'threads count to read data')
tf.app.flags.DEFINE_integer('shuffle_buffer_size',10,'buffer size for data reader')
tf.app.flags.DEFINE_integer('prefetch_buffer_size',2, 'buffer size for data reader prefetch')

#Trainer setting
tf.app.flags.DEFINE_float('learning_rate',0.00001, 'learning rate')
tf.app.flags.DEFINE_string('optimizer_type','sgd', 'optimizer to use')
tf.app.flags.DEFINE_integer('log_frequency',1000,'log frequency')
tf.app.flags.DEFINE_integer('checkpoint_frequency',10000,'check point frequency')
tf.app.flags.DEFINE_integer('max_models_to_keep',10,'max model to keep')
tf.app.flags.DEFINE_integer('early_stop_step',-1,'step to early stop, -1: disable')
tf.app.flags.DEFINE_bool('grad_mode',False,'whether to use grad mode')
tf.app.flags.DEFINE_bool('grad_float16',False,'whether to only transfer float 16')
tf.app.flags.DEFINE_bool('timeline_enable',False,'whether to record timeline')
tf.app.flags.DEFINE_string('timeline_desc','','timeline file name')
tf.app.flags.DEFINE_bool('init_status',False,'whether to print init status')
#CDSSM settings
tf.app.flags.DEFINE_string('xletter_dict','./utils/l3g.txt','xletter dictionary name')
tf.app.flags.DEFINE_integer('xletter_win_size',3,'xletter window size')
tf.app.flags.DEFINE_integer('xletter_cnt',49292, 'xletter feature num')
#Predict output setting
tf.app.flags.DEFINE_string('result_filename','predict.txt','filename of results')
tf.app.flags.DEFINE_string('result_header','query,doc','header of result file except result')
tf.app.flags.DEFINE_string('result_mode','SEQ2SEQ','output mode')
#Model specific config
tf.app.flags.DEFINE_string('bert2seq_cfg','','bert2seq config')
tf.app.flags.DEFINE_string('cdssm_cfg','','cdssm config')
tf.app.flags.DEFINE_string('bert_qk_cfg','','bert qk config')
tf.app.flags.DEFINE_string('bertpair2seq_cfg','','bertpair2seq config')
tf.app.flags.DEFINE_string('xletter2seq_cfg','','xletter2seq config')
FLAGS = tf.app.flags.FLAGS
