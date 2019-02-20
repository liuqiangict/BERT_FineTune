import random
import numpy as np
import tensorflow as tf
from param import FLAGS
from cdssm_tree_retrieve_fast import CDSSMModel
from data_reader import InputPipe
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
#from equal_groups import EqualGroupsKMeans

#def modifyLabel(label, dist):
#    majorClass = 1 if sum(label) > len(label)/2 else 0
#    minorityClass = 1 - majorClass
#    major_sample = dist[label==majorClass]
   #print(len(label), len(major_sample))
#    sorted_dist = np.sort(major_sample[:,majorClass])
#    threshold = sorted_dist[np.int(len(label)/2)-1]
#    label[(label == majorClass) & (dist[:,majorClass] > threshold)] = minorityClass

def modifyLabel(label, dist):
    majorClass = 1 if sum(label) > len(label)/2 else 0
    minorityClass = 1 - majorClass
    major_sample = dist[label==majorClass]
   #print(len(label), len(major_sample))
    sorted_dist = np.sort(major_sample[:,majorClass])
    threshold = sorted_dist[np.int(len(label)/2)-1]
    label[(label == majorClass) & (dist[:,majorClass] > threshold)] = minorityClass
    cnt = abs(sum(label)-len(label)/2)
    i = 0
    while cnt:
        if label[i] == majorClass and dist[i,majorClass] == threshold:
            cnt -= 1
            label[i] = minorityClass
        i += 1
    if not sum(label) == len(label)/2:
        print(minorityClass,sum(label),len(label))

def newSplit(vectors, startIdx, endIdx):
    #print("Start to Clustering by Kmeans...")
    kmeans = KMeans(n_clusters = 2,tol=1e-8,n_init=20,max_iter=300,verbose=0)
    vectorList = vectors[startIdx:endIdx]
    kmeans.fit(vectorList)
    label = kmeans.predict(vectorList)
    dist = kmeans.transform(vectorList)
    modifyLabel(label,dist)
    return label

def resortVec(vectors, keywords, label, startIdx, endIdx):
    tmp = []
    tmpVec = []
    tmpKwd = []
    for i, j in enumerate(label):
        tmp.append([i,j])
    tmp.sort(key=lambda x:x[1])
    for i in range(0,len(label)):
        tmpVec.append(vectors[startIdx+tmp[i][0]])
        tmpKwd.append(keywords[startIdx+tmp[i][0]])
    vectors[startIdx:endIdx] = tmpVec
    keywords[startIdx:endIdx] = tmpKwd

#Clustering setting
num_clusters = 2
#kmeans = tf.contrib.factorization.KMeans(num_clusters = num_clusters, use_mini_batch = True)
#Model setting
model = CDSSMModel()
scope = tf.get_variable_scope()
scope.reuse_variables()
saver = tf.train.Saver()

#input pipeline
input_pipe = InputPipe(FLAGS.input_validation_data_path + "/" + FLAGS.index_to_convert,FLAGS.eval_batch_size,1,1,"",False)
#input_pipe = InputPipe("../TreeRetrieve_Data/IdVec_Part0.tsv",64,1,2,"",False)
#def input_fn():
#    return input_pipe.ds
[keyword] = input_pipe.get_next()
keyword_vec = model.vector_generation(keyword, 'D')
#res = kmeans.training_graph()
#print(len(res))
#return keyword_vec
with tf.Session(config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth = True))) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(input_pipe.iterator.initializer)
    #sess.run(init_op)
    ckpt = tf.train.get_checkpoint_state(FLAGS.input_previous_model_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Load model from ", ckpt.model_checkpoint_path)
    else:
        print("No initial model found.")
    keywordList = []
    vectorList = []
    #while(True):
    i = 0
    while(True):
#    for j in range(0,20):
        if i % 1000 == 0:
            print(i,len(keywordList))
        i += 1
        try:
            keywords, vector = sess.run([keyword,keyword_vec])
            keywordList.extend(keywords)
            vectorList.extend(vector)
        except tf.errors.OutOfRangeError:
            print("End of vector generation.")
            break
    print(len(keywordList))

print("Complete the Tree...")
treeHeight = int(np.log(len(keywordList)-1)/np.log(2)) + 2
leafNode = int(pow(2,treeHeight-1))
print(treeHeight)
for i in range(0,leafNode - len(keywordList)):
    rd = random.randint(0,len(keywordList)-1)
    vectorList.append(vectorList[rd])
    keywordList.append(bytes("<UNK>" + "<"+ str(i) + ">",'utf-8'))
    #keywordList.append(keywordList[rd])
print(len(vectorList))

print("Start to clustering ...")
for i in range(1, treeHeight-1):
    node = int(pow(2,i-1))
    print("Layer",i, ":", node)
    for j in range(0,node):
        startIdx = int(leafNode / node) * j
        endIdx = int(leafNode / node) * (j + 1)
        label = newSplit(vectorList,startIdx, endIdx)
        resortVec(vectorList, keywordList, label, startIdx, endIdx)
    

#print("Start to Clustering by Kmeans...")
#kmeans = KMeans(n_clusters = 2,tol=1e-8,n_init=20,max_iter=300,verbose=0)
#kmeans.fit(vectorList)
#label = kmeans.predict(vectorList)
#dist = kmeans.transform(vectorList)
#modifyLabel(label,dist)




#print(dist.shape)
#majorClass = 1 if sum(label) > len(label)/2 else 0
#minorityClass = 1 - majorClass
#major_sample = dist[label==majorClass]
#sorted_dist = np.sort(major_sample[:,minorityClass])
#print(len(major_sample)-np.int(len(label)/2))
#threshold = sorted_dist[len(major_sample)-np.int(len(label)/2)]
#print(threshold)
#label[(label == majorClass) & (dist[:,minorityClass] < threshold)] = minorityClass
#print(len(sorted_dist))
#print(sorted_dist)
#print(sum(label))
#kmeans = KMeans(n_clusters = num_clusters)
#kmeans.fit(vectorList)
#label = kmeans.predict(vectorList)
#dist = kmeans.transform(vectorList)
#print(kmeans.inertia_)
#for i in range(0,8):
#    print(sum(label==i))
#print(sum(label==1))
#print(sum(label==2))
#clustering_model = AgglomerativeClustering(n_clusters = num_clusters,memory="./Cache")
#clustering_model.fit(vectorList)
#print(clustering_model.children_.shape)
#print(clustering_model.n_leaves_)
#print(sum(clustering_model.labels_))
#Output
print("Start to save...")
#of = open("keywordList_v2.tsv",'w')
#for i in range(0, len(keywordList)):
#    of.write(keywordList[i].decode('utf-8') + "\n")
#of.close()
#print("Keyword dictionary save done.")
if not tf.gfile.Exists(FLAGS.output_model_path):
    tf.gfile.MakeDirs(FLAGS.output_model_path)
of = tf.gfile.Open(FLAGS.output_model_path + "/" + FLAGS.tree_index_file,mode='w')
for i in range(0, len(keywordList)):
    of.write(keywordList[i].decode('utf-8') + "\n")
of.close()
print("Keyword dictionary save done.")


#clf = EqualGroupsKMeans(n_clusters=8,verbose=1)
#clf.fit(vectorList)
#for i in range(0,8):
#    print(sum(clf.labels_ == i))

#of = open("label.txt",'w')
#dist.tofile(of, sep="\n", format="%s")
#of.close()
#of = open("graph.txt",'w')
#for i in range(0,clustering_model.children_.shape[0]):
#clustering_model.children_.tofile(of, sep="\n", format="%s")
    #of.write(clustering_model.children_[i][0] + "," + clustering_model.children_[i][1] + "\n")
#of.close()

