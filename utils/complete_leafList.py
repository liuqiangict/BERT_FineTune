import tensorflow as tf
import numpy as np
import random
from param import FLAGS
keywordList = []
print("Read keyword List...")
of = tf.gfile.Open(FLAGS.input_validation_data_path + "/" + FLAGS.index_to_convert)
for i in of:
    keywordList.append(i.strip())
of.close()

print("Complete the Tree...")
treeHeight = int(np.log(len(keywordList)-1)/np.log(2)) + 2
leafNode = int(pow(2,treeHeight-1))
print("Tree height: ", treeHeight)
oriLen = len(keywordList)
for i in range(0,leafNode - oriLen):
    rd = random.randint(0,len(keywordList))
    keywordList.insert(rd, "<UNK>" + "<"+ str(i) + ">")
print("After completion:", len(keywordList))

print("Output keyword List...")
if not tf.gfile.Exists(FLAGS.output_model_path):
    tf.gfile.MakeDirs(FLAGS.output_model_path)
of = tf.gfile.Open(FLAGS.output_model_path + "/" + FLAGS.tree_index_file,mode='w')
for i in range(0, len(keywordList)):
    of.write(keywordList[i] + "\n")
of.close()
print("Keyword dictionary save done.")
