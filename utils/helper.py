import tensorflow as tf

class BaseConfig:
    def __init__(self, config_name, config_dict = None):
        self.notprint = self.not_to_print()
        self.config_name = config_name
    def init_config(self,config_dict):
        for key in self.__dict__:
            if key in config_dict:
                self.__dict__[key] = config_dict[key]
    def print_config(self):
        print(self.config_name, "config:")
        print("{"+", ".join(['"' + k + '"' + ": " + str(self.__dict__[k]) for k in self.__dict__ if k not in self.notprint])+"}")
    def not_to_print(self):
        return set(["input_training_data_path","input_validation_data_path","input_previous_model_path","output_model_path","log_dir","notprint","config_name"])

def parse_config(cfg):
    fields = cfg.strip().split(",")
    res = dict()
    for f in fields:
        if len(f) == 0:
            continue
        k,v = f.split(":")
        if not len(v) or not v[0].isdigit():
            res[k] = v
        elif "." in v:
            res[k] = float(v)
        else:
            res[k] = int(v)
    return res

def preprocess_move(path,dest,conf):
    if conf == 1: #add_unk_move
        add_unk_move(path,dest)

def add_unk_move(path,dest):
    outputter = tf.gfile.GFile(dest, mode = "w")
    #outputter.write("</s>\n<s>\n<unk>\n")
    cnt = 0
    for line in tf.gfile.GFile(path):
        if cnt == 0 and not line.strip('\ufeff').strip('\n') == '</s>':
            outputter.write("</s>\n<s>\n<unk>\n")
            cnt += 3
        outputter.write(line.strip('\ufeff'))
        cnt += 1
    outputter.close()
    print("Preprocess_move:", cnt, "words saved")

def parse_dims(dims_str):
    dims = [int(dim) for dim in dims_str.split(',')]
    return dims
