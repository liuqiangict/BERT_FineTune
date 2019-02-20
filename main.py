import os
import tensorflow as tf
import json
from models.bert2seq import Bert2Seq
from models.bertpair2seq import BertPair2Seq
from models.cdssm import CDSSM
from models.bert_qk import BertQK
from models.xletter2seq import Xletter2Seq
from config.param import FLAGS
from utils.helper import preprocess_move, parse_config
from utils.data_reader import InputPipe
from utils.trainer import SingleboxTrainer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if FLAGS.local:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i)for i in range(0,FLAGS.local)])
def get_config(): #Priority:(htol) model specific > json > FLAGS
    flag_dict = dict(FLAGS.__flags.items())
    config_dict = dict([(k,flag_dict[k].value) for k in flag_dict])
    config_json = "./config/" + FLAGS.modeltype.lower() + "_config.json"
    with tf.gfile.GFile(config_json,"r") as reader:
        text = reader.read()
        config_dict.update(json.loads(text))
    config_dict.update(parse_config(config_dict[FLAGS.modeltype.lower()+"_cfg"]))
    return config_dict

def move_file(cfg):
    if "move" not in cfg:
        return
    move_dict = parse_config(cfg["move"])
    print("files to move:", move_dict)
    for f in move_dict:
        fname = cfg[f]
        if move_dict[f] == 0:
            tf.gfile.Copy(FLAGS.input_previous_model_path + "/" + fname,FLAGS.output_model_path + "/" + fname,True)
        else:
            preprocess_move(FLAGS.input_previous_model_path + "/" + fname,FLAGS.output_model_path + "/" + fname ,move_dict[f])
    cfg["vocab_path"] = FLAGS.output_model_path


def model_selector(config_dict):
    if config_dict["modeltype"].lower() == "bert2seq":
        return Bert2Seq(config_dict)
    if config_dict["modeltype"].lower() == "cdssm":
        return CDSSM(config_dict)
    if config_dict["modeltype"].lower() == "bert_qk":
        return BertQK(config_dict)
    if config_dict["modeltype"].lower() == "bertpair2seq":
        return BertPair2Seq(config_dict)
    if config_dict["modeltype"].lower() == "xletter2seq":
        return Xletter2Seq(config_dict)

def train():
    #00.Get Config
    cfg = get_config()
    print("all config:",cfg)
    #01.Move files
    move_file(cfg)
    #02.Init Model and InputPipe
    with tf.Graph().as_default():
        model = model_selector(cfg)
        train_input_pipe = InputPipe("train",cfg)
        auc_eval_pipe, bleu_eval_pipe = None, None
        if cfg["auc_eval"]:
            auc_eval_pipe = InputPipe("eval_auc",cfg)
        if cfg["bleu_eval"]:
            bleu_eval_pipe = InputPipe("eval_bleu",cfg)
        trainer = SingleboxTrainer(model, train_input_pipe, cfg, auc_eval_pipe, bleu_eval_pipe)
        trainer.train()
    
def predict():
    cfg = get_config()
    print("all config:",cfg)
    move_file(cfg)
    outputter = tf.gfile.GFile(FLAGS.output_model_path + "/" + cfg["result_filename"], mode="w")
    pred_mode = tf.contrib.learn.ModeKeys.INFER if cfg["mode"] == "infer" else tf.contrib.learn.ModeKeys.EVAL
    with tf.Graph().as_default():
        model = model_selector(cfg)
        if pred_mode == tf.contrib.learn.ModeKeys.INFER:
            pred_pipe = InputPipe("infer",cfg)
            trainer = SingleboxTrainer(model, None, cfg, None, pred_pipe)
        else:
            pred_pipe = InputPipe("score",cfg)
            trainer = SingleboxTrainer(model, None, cfg, pred_pipe, None)
        trainer.predict(pred_mode, outputter, cfg["result_mode"],cfg["result_header"])

if __name__ == '__main__':
    #Create folders
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.output_model_path):
        tf.gfile.MakeDirs(FLAGS.output_model_path)
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'infer' or FLAGS.mode == 'score':
        predict()
    elif FLAGS.mode == 'build_graph':
        build_predict_graph()
