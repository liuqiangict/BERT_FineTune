import sys
sys.path.append('./')
import tensorflow as tf
import time
from datetime import datetime
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import nltk
from module.bert import get_assignment_map_from_checkpoint
from utils.helper import BaseConfig
from tensorflow.python.client import timeline
class TrainerConfig(BaseConfig):
    def __init__(self, config_dict, config_name='trainer'):
        super(TrainerConfig,self).__init__(config_name)
        self.mode = 'train'
        self.learning_rate = 0.01
        self.modeltype="CDSSM"
        self.optimizer_type = "sgd"
        self.auc_eval = False
        self.bleu_eval = False
        self.early_stop_step = -1
        self.metrics_early_stop = 'no'
        self.log_frequency = 1000
        self.checkpoint_frequency = 10000
        self.max_models_to_keep = 10
        self.input_previous_model_path = ''
        self.output_model_path = ''
        self.log_dir = ''
        self.grad_mode = False
        self.grad_float16 = False
        self.timeline_enable=False
        self.timeline_desc = ""
        self.init_status = False
        self.init_config(config_dict)
        self.print_config()

class Metrics:
    def __init__(self):
        self.best_value = 0
        self.best_step = 0
        self.bad_step = 0
        self.improved = False
        self.earlystop = False
        self._top = []
    def update(self, value, step, early_stop_steps):
        if value > self.best_value:
            self.best_value = value
            self.best_step = step
            self.improved = True
            self.bad_step = 0
        else:
            self.improved = False
            self.bad_step += 1
            if early_stop_steps > 0 and self.bad_step > early_stop_steps:
                self.earlystop = True

class SingleboxTrainer:
    def __init__(self, model, inp, cfg, score_inp=None, infer_inp=None):
        self.cfg = TrainerConfig(cfg)
        self.model = model
        self.devices = self.get_devices()
        self.inp = inp
        self.score_inp = score_inp
        self.infer_inp = infer_inp
        self.session = None
        self.saver = None
        self.weight_record = 0
        #tvars =  tf.trainable_variables()
        #print('before train graph', len(tvars))
        if self.cfg.mode == 'train':
            self.build_train_graph(self.cfg.grad_mode)
        #tvars =  tf.trainable_variables()
        #print('after train graph', len(tvars))
        if self.cfg.mode == 'train' and self.cfg.auc_eval or self.cfg.mode == 'score':
            self.build_eval_graph()
        if self.cfg.mode == 'train' and self.cfg.bleu_eval or self.cfg.mode == 'infer':
            self.build_infer_graph()
        #tvars =  tf.trainable_variables()
        #print('after eval graph', len(tvars))

    def build_train_graph(self,grad_mode):
        self.eval_metrics = Metrics()
        self.global_step = tf.train.get_or_create_global_step()
        self.inc_step = tf.assign_add(self.global_step,1)
        # Training progress record
        self.total_weight = [tf.Variable(0., trainable=False) for i in range(0, self.model.loss_cnt)]
        self.total_loss = [tf.Variable(0., trainable=False) for i in range(0, self.model.loss_cnt)]
        # Optimizer
        self.opts = self.model.get_optimizer(self.cfg.optimizer_type, self.cfg.learning_rate)
        if grad_mode:
            self.build_grad_train_graph()
        else:
            self.build_param_train_graph()
    
    def build_grad_train_graph(self):
        tower_grads = []
        tower_loss = [[] for i in range(0,self.model.loss_cnt)]
        for i in range(0,len(self.devices)):
            with tf.device(self.devices[i]):
                scope_name = 'device_%d' % i
                with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
                    batch_input = self.inp.get_next()
                    loss,weight = self.tower_loss(scope, batch_input)
                    v = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope_name)
                    grads = []
                    for opt in self.opts:
                        gd = opt.compute_gradients(loss[0],v)
                        if self.cfg.grad_float16:
                            grads.extend(self.cast_grads(gd,tf.float16))
                        else:
                            grads.extend(gd)
                        #grads.extend(opt.compute_gradients(loss[0]))
                    tower_grads.append(grads)
                    for j in range(0,len(loss)):
                        tower_loss[j].append((loss[j],weight))
        self.avg_loss = [self.update_loss(tower_loss[i],i) for i in range(0,len(tower_loss))]
        grads = self.sum_gradients(tower_grads,len(self.devices))
        self.train_op = []
        for i in range(0,len(self.devices)):
            if self.cfg.grad_float16:
                grads[i] = self.cast_grads(grads[i],tf.float32)
            self.train_op += [opt.apply_gradients(grads[i]) for opt in self.opts]
        #self.train_op = [opt.apply_gradients(grads) for opt in self.opts]

    
    def build_param_train_graph(self):
        # training
        tower_grads = []
        tower_loss = [[] for i in range(0,self.model.loss_cnt)]
        #Training
        for i in range(0,len(self.devices)):
            with tf.device(self.devices[i]):
                #with tf.name_scope('device_%d' % i) as scope:
                with tf.variable_scope(tf.get_variable_scope()), tf.name_scope('device_%d' % i) as scope:
                    batch_input = self.inp.get_next()
                    loss,weight = self.tower_loss(scope, batch_input)
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    grads = []
                    for opt in self.opts:
                        grads.extend(opt.compute_gradients(loss[0]))
                    tower_grads.append(grads)
                    tf.get_variable_scope().reuse_variables()
                    for j in range(0,len(loss)):
                        tower_loss[j].append((loss[j],weight))
        self.avg_loss = [self.update_loss(tower_loss[i],i) for i in range(0, len(tower_loss))]
        grads = self.sum_gradients(tower_grads,1)[0]
        self.train_op = [opt.apply_gradients(grads) for opt in self.opts]
    def build_eval_graph(self):
        tower_score = []
        if self.cfg.grad_mode:
            for i in range(0,len(self.devices)):
                with tf.device(self.devices[i]):
                    with tf.variable_scope('device_%d' % i) as scope:
                        eval_test_batch = self.score_inp.get_next()
                        score = self.tower_score(eval_test_batch)
                        tower_score.append([eval_test_batch,score])
        else:
            for i in range(0, len(self.devices)):
                with tf.device(self.devices[i]):
                    eval_test_batch = self.score_inp.get_next()
                    score = self.tower_score(eval_test_batch)
                    tower_score.append([eval_test_batch,score])
        self.score_list = self.merge_score_res(tower_score)  
        
    def build_infer_graph(self):
        tower_infer = []
        if self.cfg.grad_mode:
            for i in range(0, len(self.devices)):
                    with tf.device(self.devices[i]):
                        with tf.variable_scope('device_%d' % i) as scope:
                            infer_batch = self.infer_inp.get_next()
                            infer_res = self.tower_inference(infer_batch)
                            tower_infer.append([infer_batch, infer_res])
        else:
            for i in range(0, len(self.devices)):
                with tf.device(self.devices[i]):
                    infer_batch = self.infer_inp.get_next()
                    infer_res = self.tower_inference(infer_batch)
                    tower_infer.append([infer_batch, infer_res])
        self.infer_list = self.merge_infer_res(tower_infer)

    def update_loss(self, tower_loss, idx):
        loss, weight = zip(*tower_loss)
        loss_inc = tf.assign_add(self.total_loss[idx], tf.reduce_sum(loss) / 10000)
        weight_inc = tf.assign_add(self.total_weight[idx], tf.cast(tf.reduce_sum(weight) / 10000, tf.float32))
        avg_loss = loss_inc / weight_inc
        tf.summary.scalar("avg_loss" + str(idx), avg_loss)
        return avg_loss

    def tower_score(self, batch_input):
        inference_output = self.model.inference(batch_input, tf.contrib.learn.ModeKeys.EVAL)
        prediction = self.model.calc_score(inference_output)
        return prediction

    def tower_inference(self, batch_input):
        inference_output = self.model.inference(batch_input, tf.contrib.learn.ModeKeys.INFER)
        rewrite, seq_length = self.model.lookup_infer(inference_output)
        score = self.model.calc_score(inference_output)
        return rewrite, seq_length, score


    def tower_loss(self, scope, batch_input):
        inference_output = self.model.inference(batch_input,tf.contrib.learn.ModeKeys.TRAIN)
        loss,weight = self.model.calc_loss(inference_output)
        tf.summary.scalar("losses",loss[0])
        #losses = tf.get_collection('losses',scope)
        #total_loss = tf.add_n(losses, name='total_loss')
        return loss,weight
    
    #def merge_eval_res(self, tower_pred):
    #    test_batch, score = zip(*tower_pred)
    #    merge_batch = []
    #    for i in zip(*test_batch):
    #        if not isinstance(i[0],tf.Tensor):
    #            merge_batch.append(tf.concat([j[0] for j in i], axis = 0))
    #        else:
    #            merge_batch.append(tf.concat(i, axis=0))
        #print(merge_batch)
    #    merge_score = tf.concat(score, axis = 0)
    #    return merge_batch, merge_score
    
    def merge_score_res(self, tower_score):
        score_batch, score_res = zip(*tower_score)
        merge_batch = {}
        for k in score_batch[0]:
            if isinstance(score_batch[0][k], tf.Tensor):
                merge_batch[k] = tf.concat([score_batch[i][k] for i in range(0,len(score_batch))],axis=0)
            else:
                merge_batch[k] = tf.concat([score_batch[i][k][0] for i in range(0,len(score_batch))],axis=0)
        merge_score = tf.concat(score_res, axis=0)
        return merge_batch, merge_score

    #def merge_infer_res(self, tower_infer):
    #    infer_batch, infer_res = zip(*tower_infer)
    #    merge_batch = []
    #    merge_res = []
    #    for i in zip(*infer_batch):
    #        if not isinstance(i[0],tf.Tensor):
    #            merge_batch.append(tf.concat([j[0] for j in i], axis = 0))
    #        else:
    #            merge_batch.append(tf.concat(i, axis=0))
    #    for i in zip(*infer_res):
    #        merge_res.append(tf.concat(i,axis=0))
    #    return merge_batch, merge_res

    def merge_infer_res(self, tower_infer):
        infer_batch, infer_res = zip(*tower_infer)
        merge_batch = {}
        merge_res = []
        for k in infer_batch[0]:
            if isinstance(infer_batch[0][k], tf.Tensor):
                merge_batch[k] = tf.concat([infer_batch[i][k] for i in range(0,len(infer_batch))],axis=0)
            else:
                merge_batch[k] = tf.concat([infer_batch[i][k][0] for i in range(0,len(infer_batch))],axis=0)
        for i in zip(*infer_res):
            merge_res.append(tf.concat(i,axis=0))
        return merge_batch, merge_res
    
    #def merge_search_res(self, tower_search):
    #    search_batch, search_res = zip(*tower_search)
    #    merge_batch = []
    #    for i in zip( *search_batch):
    #        if not isinstance(i[0], tf.Tensor):
    #            merge_batch.append(tf.concat([j[0] for j in i], axis = 0))
    #        else:
    #            merge_batch.append(tf.concat(i, axis=0))
    #    merge_res = tf.concat(search_res, axis=0)
    #    return merge_batch, merge_res
    def merge_search_res(self, tower_search):
        search_batch, search_res = zip(*tower_search)
        merge_batch = {}
        for k in search_batch[0]:
            if isinstance(search_batch[0][k], tf.Tensor):
                merge_batch[k] = tf.concat([search_batch[i][k] for i in range(0,len(search_batch))],axis=0)
            else:
                merge_batch[k] = tf.concat([search_batch[i][k][0] for i in range(0,len(search_batch))],axis=0)
        merge_res = tf.concat(search_res, axis=0)
        return merge_batch, merge_res
    
    def sum_gradients(self, tower_grads, grad_replica):
        sum_grads = [[] for i in range(0,grad_replica)]
        #sum_grads = []
        for grad_and_vars in zip(*tower_grads):
            if isinstance(grad_and_vars[0][0],tf.Tensor):
               #print(grad_and_vars[0][0])
                grads = []
                for g, _ in grad_and_vars:
                    expanded_g = tf.expand_dims(g,0)
                    grads.append(expanded_g)
                grad = tf.concat(grads, 0)
                grad = tf.reduce_sum(grad, 0)
                for i in range(0,grad_replica):
                    sum_grads[i] += [(grad,grad_and_vars[i][1])]
                #sum_grads += [(grad,grad_and_vars[i][1]) for i in range(0,grad_replica)]
               #v = grad_and_vars[0][1]
               #grad_and_var = (grad, v)
               #sum_grads.append(grad_and_var)
            else:
                values = tf.concat([g.values for g,_ in grad_and_vars],0)
                indices = tf.concat([g.indices for g,_ in grad_and_vars],0)
                #merge same indexes
                if self.cfg.grad_mode and grad_replica > 1:
                    indices, idx = tf.unique(indices)
                    values = tf.math.unsorted_segment_sum(values, idx, tf.size(indices))
                for i in range(0,grad_replica):
                    sum_grads[i] += [(tf.IndexedSlices(values, indices), grad_and_vars[i][1])]
               #v = grad_and_vars[0][1]
               #grad_and_var = (tf.IndexedSlices(values, indices),v)
               #sum_grads.append(grad_and_var)
        return sum_grads
    #Add for grad mode
    def cast_grads(self, grad_and_vars, type):
        merge_grads = []
        for grad_and_var in grad_and_vars: 
            g, v = grad_and_var
            if isinstance(g,tf.Tensor):
                g_v = (tf.cast(g, type), v)
                merge_grads.append(g_v)
            else:
                values = tf.cast(g.values, type)
                indices = g.indices
                g_v = (tf.IndexedSlices(values, indices),v)
                merge_grads.append(g_v)
        return merge_grads
    def init_sync(self):
        """make model same per gpu"""
        assignment_map = {}
        tvars =  tf.trainable_variables()
        #print(tvars)
        for var in tvars:
            di , name = int(var.name[7:8]), var.name[9:]
            if di == 0:
                assignment_map[name] = var
                
        for var in tvars:
            di , name = int(var.name[7:8]), var.name[9:]
            if di == 0:
                continue
            tf.assign(var, assignment_map[name])
            tf.add_to_collection('assignOps', var.assign(assignment_map[name]))
        return tf.get_collection('assignOps')
    
    def Saveable_Variables(self):
        v = []
        for var in tf.global_variables():
            saveable = True
            for i in range(1, len(self.devices)):
                if 'device_' + str(i) in var.name:
                    saveable = False
                    break
            if saveable:
                v.append(var)
        return v
    def get_devices(self):
        devices = []
        if os.environ and 'CUDA_VISIBLE_DEVICES' in os.environ:
            for i, gpu_id in enumerate(os.environ['CUDA_VISIBLE_DEVICES'].split(',')):
                gpu_id = int(gpu_id)
                if gpu_id < 0:
                    continue
                devices.append('/gpu:'+str(gpu_id))
        if not len(devices):
            devices.append('/cpu:0')
        print("available devices", devices)
        return devices
    def init_ops(self):
        if self.cfg.mode == 'train':
            ops = [tf.local_variables_initializer(),tf.tables_initializer(),tf.global_variables_initializer(),self.inp.iterator.initializer]
        else:
            ops = [tf.local_variables_initializer(),tf.tables_initializer(),tf.global_variables_initializer()]
        return ops
    def train_ops(self):
        return [self.train_op, self.avg_loss, self.total_weight, self.inc_step]
    def scoring_ops(self):
        return self.score_list
    def inference_ops(self):
        return self.infer_list#[self.rewrite, self.inference_res_bleu]
    def search_ops(self):
        return self.search_list
    def improved(self):
        return self.eval_metrics.improved
    def early_stop(self):
        return self.eval_metrics.earlystop
    def metrics_update(self, score, step):
        self.eval_metrics.update(score, step, self.cfg.early_stop_step)
    def print_log(self, total_weight, step, avg_loss):
        examples, self.weight_record = total_weight[0] - self.weight_record, total_weight[0]
        current_time = time.time()
        duration, self.start_time = current_time - self.start_time, time.time()
        examples_per_sec = examples * 10000 / duration
        sec_per_steps = float(duration / self.cfg.log_frequency)
        format_str = "%s: step %d, %5.1f examples/sec, %.3f sec/step, %.1f samples processed,"
        #print(format_str % (datetime.now(), step, avg_loss, examples_per_sec, sec_per_steps, total_weight)) 
        avgloss_str = "avg_loss = " + ",".join([str(avg_loss[i]) for i in range(0,len(avg_loss))])
        print(format_str % (datetime.now(), step, examples_per_sec, sec_per_steps, total_weight[0]) + avgloss_str)
        
    def eval(self, step,eval_type):
        #eval_pipe.reset()
        imporved_mark = ""
        if eval_type == "auc":
            self.session.run([self.score_inp.iterator.initializer])
            score_list, label_list = [],[]
            while True:
                try:
                    input_batch, score = self.session.run(self.scoring_ops())
                    #print(input_batch)
                    label_list.extend([int(i) for i in input_batch["label"]])
                    score_list.extend(score)
                except tf.errors.OutOfRangeError:
                    print("auc_evaluation done.")
                    break
            auc_score = roc_auc_score(label_list, score_list)
            improved_mark = ""
            if self.cfg.metrics_early_stop == 'auc':
                self.metrics_update(auc_score, step)
                improved_mark = "?" if self.improved() else ""
            format_str = "%s: step %d, %d sample evaluated, eval_auc = %.10f" + improved_mark
            print(format_str % (datetime.now(), step, len(score_list), auc_score))
        else:
            self.session.run([self.infer_inp.iterator.initializer])
            bleu_score = []
            noscore = 0
            while True:
                try:
                #if 1:
                    input_batch, [rewrite,resLen,score] = self.session.run(self.inference_ops())
                    for i in range(len(rewrite)):
                        rwt = " ".join([rewrite[i][j].decode('utf-8') for j in range(0, resLen[i]-1)])
                        try:
                            #print(input_batch["doc"][i], rwt)
                            bleu_score.append(nltk.translate.bleu_score.sentence_bleu(input_batch["doc"][i].decode('utf-8').split(";"), rwt))
                        except:
                            noscore += 1
                except tf.errors.OutOfRangeError:
                    print("bleu evaluation done.")
                    break
            bleu_score_avg = np.mean(bleu_score)
            improved_mark = ""
            if self.cfg.metrics_early_stop == 'bleu':
                self.metrics_update(bleu_score_avg, step)
                improved_mark = "?" if self.improved() else ""
            format_str = "%s: step %d, %d sample evaluated, %d sample without score, eval_bleu = %.10f" + improved_mark
            print(format_str % (datetime.now(), step, len(bleu_score), noscore, bleu_score_avg))


    def initsession_and_loadmodel(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement=True
        scope = tf.get_variable_scope()
        scope.reuse_variables()
        if self.cfg.grad_mode:
            self.saver = tf.train.Saver(var_list=self.Saveable_Variables(), max_to_keep=self.cfg.max_models_to_keep,name='model_saver')
        else:
            self.saver = tf.train.Saver(max_to_keep=self.cfg.max_models_to_keep,name='model_saver')
        ckpt = tf.train.get_checkpoint_state(self.cfg.input_previous_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            tvars = tf.trainable_variables()
            #print(tvars)
            assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, ckpt.model_checkpoint_path)
            #print(assignment_map)
            tf.train.init_from_checkpoint(ckpt.model_checkpoint_path,assignment_map)
            print("Load model from ", ckpt.model_checkpoint_path)
        else:
            print("No Initial Model Found.")
        if self.cfg.init_status:
            print("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                print("  name = "+var.name+" shape = ",var.shape, init_string)
        self.session = tf.Session(config=config)
        self.session.run(self.init_ops())
        if self.cfg.grad_mode:
            self.session.run(self.init_sync())
    
    def train(self):
        self.initsession_and_loadmodel()
        summ_writer = tf.summary.FileWriter(self.cfg.log_dir, self.session.graph)
        summary_op = tf.summary.merge_all()
        self.start_time = time.time()
        if self.cfg.timeline_enable:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        def eval_once(step):
            if self.cfg.auc_eval:
                self.eval(step,'auc')
            if self.cfg.bleu_eval:
                self.eval(step,'bleu')
        step = 0
        eval_once(step)
        while True:
            try:
                #eval first
                if self.cfg.timeline_enable:
                    _,avg_loss,total_weight,step,summary = self.session.run(self.train_ops() + [summary_op], options = options, run_metadata = run_metadata)
                else:
                    _,avg_loss,total_weight,step,summary = self.session.run(self.train_ops() + [summary_op])
                if step % self.cfg.log_frequency == 1:
                    if self.cfg.timeline_enable:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with tf.gfile.Open(self.cfg.log_dir+'//timeline_'+self.cfg.timeline_desc+'_step_%d.json' % step, mode='w') as f:
                            f.write(chrome_trace)
                    summ_writer.add_summary(summary,step)
                    self.print_log(total_weight,step,avg_loss)
                if step % self.cfg.checkpoint_frequency == 0:
                    eval_once(step)
                    if self.improved():
                        self.saver.save(self.session, self.cfg.output_model_path + "/" + self.cfg.modeltype + "_model", global_step=step)
                    elif self.early_stop():
                        print("\nEarly stop")
                        break
            except tf.errors.OutOfRangeError:
                print("End of training.")
                break
        eval_once(step)
        if self.improved():
            self.saver.save(self.session, self.cfg.output_model_path + "/" + self.cfg.modeltype + "_model_final", global_step=step)


    def predict(self, predict_mode, outputter, output_mode, output_header):
        self.initsession_and_loadmodel()
        header = output_header.split(",")
        if predict_mode == tf.contrib.learn.ModeKeys.EVAL:
            self.session.run([self.score_inp.iterator.initializer])
            while True:
                try:
                    input_batch, score = self.session.run(self.scoring_ops())
                    for i in range(len(score)):
                        output_str = "\t".join(input_batch[h][i].decode('utf-8') for h in header)
                        output_str += "\t" + str(score[i])
                    outputter.write(output_str+"\n")
                except tf.errors.OutOfRangeError:
                    print("score predict done.")
                    break
        else:
            self.session.run([self.infer_inp.iterator.initializer])
            while True:
                try:
                #if 1:
                    input_batch, [rewrite, resLen, score] = self.session.run(self.inference_ops())
                    #print(input_batch)
                    for i in range(0,len(resLen)):
                        output_str = "\t".join(input_batch[h][i].decode('utf-8') for h in header) 
                        if output_mode == "TREE": # tree based retrieval
                            output_str += "\t" + ",".join([rewrite[i][j].decode('utf-8') for j in range(0, resLen[i] - 1)])
                            output_str += "\t" + ",".join([str(score[i][j]) for j in range(0, resLen[i] - 1)])
                        elif output_mode == "VECTOR":
                            output_str += "\t" + ",".join([str(rewrite[i][j]) for j in range(0, resLen[i])])
                        else:
                            if self.model.cfg.beam_width <= 1:
                                output_str += "\t" + " ".join([rewrite[i][j].decode('utf-8') for j in range(0, resLen[i] - 1)])
                                output_str += "\t" + str(score[i])
                            else:
                                res = []
                                tmpLen = ""
                                for k in range(0, self.model.cfg.beam_width):
                                    tmpRes = " ".join([rewrite[i][j][k].decode('utf-8') for j in range(0,resLen[i][k]-1)])
                                    #tmpRes = " ".join([rewrite[i][j][k].decode('utf-8') for j in range(0,20)])
                                    #tmpRes = " ".join([str(rewrite[i][j][k]) for j in range(0,resLen[i][k]-1)])
                                    tmpScore = score[i][k]
                                    tmpLen += " " + str(resLen[i][k])
                                    res.append((tmpScore,tmpRes))
                                #res.sort(reverse=True)
                                output_str += "\t" + "\t".join( y + "\t" + str(x) for x,y in res)
                                #output_str += "\t" + "\t".join( y for x,y in res) + "\t" + tmpLen + "\t" + str(len(resLen[i]))
                        outputter.write(output_str + "\n")
                except tf.errors.OutOfRangeError:
                    print("inference done.")
                    break
                except:
                    print(input_batch["query"],input_batch["doc"])

    def search(self, outputter, outheader):
        header = outheader.split(',')
        cnt = 0
        self.session.run([self.infer_inp.iterator.initializer])
        while True:
            try:
                cnt += 1
                if cnt % 10000 == 0:
                    print(cnt)
                input_batch,res = self.session.run(self.search_ops())
                #print(input_batch)
                #print(res[0])
                for i in range(len(res)):
                    output_str = "\t".join(input_batch[h].decode('utf-8') for h in header)
                    output_str += "\t" + ",".join([res[i][j].decode('utf-8') for j in range(0, len(res[i]))])
                    outputter.write(output_str + "\n")
            except tf.errors.OutOfRangeError:
                print("search done.")
                break

