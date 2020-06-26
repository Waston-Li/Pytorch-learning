import numpy as np  
import tensorflow as tf

def BP_Name_Net(input_data, labels_name, class_num):
    name_hidden_weight = tf.Variable(tf.truncated_normal(shape = [1024,4], dtype = tf.float64, stddev = 0.01))
    name_hidden_bias = tf.Variable(tf.truncated_normal(shape = [4], dtype = tf.float64, stddev = 0.01))
    name_hidden_opt = tf.nn.relu(tf.matmul(input_data,name_hidden_weight) + name_hidden_bias)
    name_finnal_weight = tf.Variable(tf.truncated_normal(shape = [4, 11],dtype = tf.float64, stddev = 0.01))
    name_finnal_bias = tf.Variable(tf.truncated_normal(shape = [11],dtype = tf.float64, stddev = 0.01))
    name_finnal_opt = tf.nn.relu(tf.matmul(name_hidden_opt, name_finnal_weight) + name_finnal_bias)

    return name_finnal_opt

def BP_Dir_Net(input_data, labels_name, class_num):
    dir_hidden_weight = tf.Variable(tf.truncated_normal(shape = [1024, 2], dtype = tf.float64, stddev = 0.01))
    dir_hidden_bias = tf.Variable(tf.truncated_normal(shape = [2], dtype = tf.float64, stddev = 0.01))
    dir_hidden_opt = tf.nn.relu(tf.matmul(input_data,dir_hidden_weight) + dir_hidden_bias)
    dir_finnal_weight = tf.Variable(tf.truncated_normal(shape = [2,class_num[labels_name]],dtype = tf.float64, stddev = 0.01))
    dir_finnal_bias = tf.Variable(tf.truncated_normal(shape = [4],dtype = tf.float64, stddev = 0.01))
    dir_finnal_opt = tf.nn.relu(tf.matmul(dir_hidden_opt,dir_finnal_weight)+dir_finnal_bias)

    return dir_finnal_opt


    