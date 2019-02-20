import sys
import tensorflow as tf
from utils.param import FLAGS
from utils.xletter import XletterPreprocessor
from utils.layers import xletter_feature_extractor, mask_maxpool
import math
from tensorflow.python.ops import lookup_ops
if FLAGS.use_mstf_ops == 1:
    import tensorflow.contrib.microsoft as mstf

def parse_dims(dims_str):
    dims = [int(dim) for dim in dims_str.split(',')]
    return dims
def count_idx(filename):
    count = 0
    #for line in open(filename,encoding='utf-8'):
    for line in tf.gfile.GFile(filename):
        count += 1
    return count
def default_init():
    return tf.contrib.layers.variance_scaling_nitializer(factor=1.0, mode = 'FAN_AVG', uniform = True)

class TreeModel():
    def __init__(self):
        TreeHeight = lambda x: int(math.log(x-1)/math.log(2)) + 2
        indexCnt = count_idx(FLAGS.input_previous_model_path + "/" + FLAGS.tree_index_file)
        self.tree_height = TreeHeight(indexCnt+1)
        self.tree_index = lookup_ops.index_table_from_file(FLAGS.input_previous_model_path + "/" + FLAGS.tree_index_file, default_value = indexCnt)
        self.reverse_tree_index = lookup_ops.index_to_string_table_from_file(FLAGS.input_previous_model_path + "/" + FLAGS.tree_index_file, default_value = '<unk>')
        self.dims = parse_dims(FLAGS.semantic_model_dims)
        self.layer_embedding = tf.get_variable(name='tree_node_emb', shape = [pow(2, self.tree_height -1) ,self.dims[-1]])
        if not FLAGS.leaf_content_emb:
            self.leaf_embedding = tf.get_variable(name='leaf_node_emb', shape = [pow(2, self.tree_height -1) ,self.dims[-1]])
        if FLAGS.use_mstf_ops == 1:
            self.op_dict = mstf.dssm_dict(FLAGS.xletter_dict)
        elif FLAGS.use_mstf_ops == -1:
            self.op_dict = XletterPreprocessor(FLAGS.xletter_dict, FLAGS.xletter_win_size)
        else:
            self.op_dict = None

    def inference(self, input_fields, mode):
        if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
            if FLAGS.use_mstf_ops:
                query, doc = input_fields[0], input_fields[1]
            else:
                query, doc = input_fields[0][0], input_fields[1][0]
            query_vec = self.vector_generation(query, 'Q')
            doc_id = self.tree_index.lookup(doc)
            doc_vec = self.vector_generation(doc, 'D', doc_id)
            return query_vec, doc_vec, doc_id
        elif mode == tf.contrib.learn.ModeKeys.INFER:
            if FLAGS.use_mstf_ops:
                query = input_fields[0]
            else:
                query = input_fields[0][0]
            query_vec = self.vector_generation(query,'Q')
            return [query_vec] + self.search(query_vec)
    
    def search(self, query_vec):
        #[batch_size,vec_dim]
        batch_size = tf.shape(query_vec)[0]
        query_vec = tf.expand_dims(query_vec, axis = 1)
        start_layer = math.floor(math.log(FLAGS.top_k)/math.log(2)) + 2
        #[batch_size,N]
        top_index = tf.tile(tf.expand_dims(tf.range(pow(2,start_layer-2), pow(2,start_layer-1)),axis=0),[batch_size,1])
        #top_index = tf.range(pow(2, start_layer-1),pow(2,start_layer))
        #return top_index
        for i in range(start_layer, self.tree_height):
            #[batch_size,2N]
            eval_index = tf.concat([tf.cast(top_index * 2,tf.int32), tf.cast(top_index * 2 + 1,tf.int32)],axis = 1)
            #return eval_index
            #[batch_size,2N,vec_dim]
            eval_emb = tf.gather(self.layer_embedding, eval_index)
            #return tf.shape(eval_emb)
            eval_emb = tf.nn.l2_normalize(eval_emb, dim = 2)
            eval_emb_transpose = tf.transpose(eval_emb,[0,2,1])
            #return tf.shape(query_vec)
            #[batch_size,2N] hope so....
            eval_score = tf.matmul(query_vec, eval_emb_transpose)
            eval_score = tf.squeeze(eval_score)
            #return eval_score
            #return tf.shape(eval_score)
            #Select Top N
            values, top_index = tf.nn.top_k(eval_score,FLAGS.top_k, False)
            top_index = tf.reshape(top_index,[-1,FLAGS.top_k])
            #top_index = tf.expand_dims(top_index, axis=2)
            batch_id = tf.tile(tf.expand_dims(tf.range(batch_size),axis=1),[1,tf.shape(top_index)[1]])
            expand_index = tf.concat([tf.expand_dims(batch_id,axis=2),tf.expand_dims(top_index,axis=2)],axis=-1)
            #return top_index, batch_id ,expand_index
            top_index = tf.gather_nd(eval_index,expand_index)
            #return top_index,eval_index, what
        res = top_index * 2 - pow(2, self.tree_height - 1)
        res = tf.concat([res, res+1],axis=1)
        values = tf.concat([values, values], axis=1)
        if not FLAGS.leaf_content_emb:
            eval_emb = tf.gather(self.leaf_embedding, tf.cast(res - pow(2, self.tree_height - 1),tf.int32))
            eval_emb = tf.nn.l2_normalize(eval_emb, dim = 2)
            eval_emb_transpose = tf.transpose(eval_emb,[0,2,1])
            eval_score = tf.matmul(query_vec, eval_emb_transpose)
            eval_score = tf.squeeze(eval_score)
            values, top_index = tf.nn.top_k(eval_score,FLAGS.top_k, False)
            top_index = tf.reshape(top_index,[-1,FLAGS.top_k])
            batch_id = tf.tile(tf.expand_dims(tf.range(batch_size),axis=1),[1,tf.shape(top_index)[1]])
            expand_index = tf.concat([tf.expand_dims(batch_id,axis=2),tf.expand_dims(top_index,axis=2)],axis=-1)
            #return top_index, batch_id ,expand_index
            res = tf.gather_nd(res,expand_index)
        return [res,values] 

    def vector_generation(self, text, model_prefix, doc_id = None):
        if model_prefix == "D" and not FLAGS.leaf_content_emb:
            return tf.nn.l2_normalize(tf.gather(self.leaf_embedding, doc_id),dim = 1)
        dims = parse_dims(FLAGS.semantic_model_dims)
        text_vecs, step_mask, sequence_length = xletter_feature_extractor(text, model_prefix, self.op_dict, FLAGS.xletter_cnt, FLAGS.xletter_win_size, FLAGS.dim_xletter_emb)
        maxpooling_vec = mask_maxpool(tf.nn.tanh(text_vecs),step_mask)
        dim_input = FLAGS.dim_xletter_emb
        input_vec = maxpooling_vec
        for i, dim in enumerate(self.dims):
            dim_output = dim
            random_range = math.sqrt(6.0/(dim_input+dim_output))
            with tf.variable_scope("semantic_layer{:}".format(i),reuse=tf.AUTO_REUSE):
                weight = tf.get_variable("weight_" + model_prefix, shape = [dim_input, dim_output], initializer = tf.random_uniform_initializer(-random_range, random_range))
                output_vec = tf.matmul(input_vec, weight)
                output_vec = tf.nn.tanh(output_vec)
                input_vec = output_vec
        normalized_vec = tf.nn.l2_normalize(output_vec, dim = 1)
        return normalized_vec

    def calc_loss(self, inference_res):
        query_vec, doc_vec, doc_id = inference_res
        batch_size = tf.shape(query_vec)[0]
        #Leaf Layer Loss
        posCos = tf.reduce_sum(tf.multiply(query_vec, doc_vec), axis = 1)
        allCos = [posCos]
        for i in range(0, FLAGS.negative_sample):
            random_indices = (tf.range(batch_size) + tf.random_uniform([batch_size],1,batch_size, tf.int32)) % batch_size
            negCos = tf.reduce_sum(tf.multiply(query_vec, tf.gather(doc_vec, random_indices)),axis=1)
            allCos.append(tf.where(tf.equal(negCos,1),tf.zeros_like(negCos),negCos))
        allCos = tf.stack(allCos, axis=1)
        softmax = tf.nn.softmax(allCos * FLAGS.softmax_gamma, dim=1)
        leafLoss = tf.reduce_sum(-tf.log(softmax[:,0]))

        #Node Layer Loss
        doc_idx = doc_id + pow(2, self.tree_height - 1)
        nodeLoss = tf.zeros_like(leafLoss,dtype=tf.float32)
        for i in range(4, self.tree_height):
            cosines = []
            posIdx = tf.cast(doc_idx // pow(2, self.tree_height - i),tf.int32)
            posVec = tf.nn.l2_normalize(tf.gather(self.layer_embedding, posIdx),dim = 1)
            cosines.append(tf.reduce_sum(tf.multiply(query_vec, posVec), axis = 1))
            nodeCnt = pow(2,i-1)
            for j in range(0, FLAGS.layer_negative_sample):
                random_idx = nodeCnt + (posIdx - nodeCnt + tf.random_uniform([batch_size],1,nodeCnt,tf.int32)) % nodeCnt
                cosines.append(tf.reduce_sum(tf.multiply(query_vec, tf.nn.l2_normalize(tf.gather(self.layer_embedding, random_idx),dim=1)),axis=1))
            cosines = tf.stack(cosines, axis=1)
            softmax = tf.nn.softmax(cosines * FLAGS.softmax_gamma, dim = 1)
            nodeLoss += tf.reduce_sum(-tf.log(softmax[:,0]))

        weight = batch_size
        loss = tf.cast(leafLoss + FLAGS.layer_weight * nodeLoss,tf.float32)
        tf.summary.scalar('softmax_losses',loss)
        return [loss, leafLoss, nodeLoss], weight

    def calc_score(self, inference_res):
        if FLAGS.mode == 'predict':
            query_vec, search_res, score = inference_res
        else:
            query_vec, doc_vec, doc_id = inference_res
            score = tf.reduce_sum(tf.multiply(query_vec, doc_vec), axis = 1)
        return score

    def get_optimizer(self):
        return [tf.train.GradientDescentOptimizer(FLAGS.learning_rate)]
   
    def lookup_infer(self, inference_res):
        query_vec, search_res, search_values = inference_res
        if FLAGS.leaf_content_emb:
            seq_len = 2 * FLAGS.top_k
        else:
            seq_len = FLAGS.top_k
        return self.reverse_tree_index.lookup(tf.to_int64(search_res)),tf.ones([tf.shape(query_vec)[0]],dtype=tf.int32)+seq_len


if __name__ == '__main__':
    from data_reader import InputPipe 
    m = CDSSMModel()
    pred_pipe = InputPipe(FLAGS.input_validation_data_path + "/bleu_data.txt", FLAGS.eval_batch_size,1,2,"",True)
    query,keyword = pred_pipe.get_next()
    output = m.search(query)
    with tf.Session() as sess:
        scope = tf.get_variable_scope()
        scope.reuse_variables()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(pred_pipe.iterator.initializer)
        ckpt = tf.train.get_checkpoint_state(FLAGS.input_previous_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Load model from ", ckpt.model_checkpoint_path)
        else:
            print("No initial model found.")
        print(sess.run([query,output]))
