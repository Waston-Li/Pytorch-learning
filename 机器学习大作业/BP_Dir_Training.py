import tensorflow as tf
import BPModule
import os
import numpy as np
import DataProcessModule

data_dir = "/media/test/B/python_文件/机器学习大作业"
save_dir = "/media/test/B/python_文件/机器学习大作业/ckpt/BP"
train_filename = os.path.join(data_dir, "train")
test_filename = os.path.join(data_dir, "test")
train_data = DataProcessModule.ImageData(train_filename, True, True)
print("train_data load success")
test_data = DataProcessModule.ImageData(test_filename, True, True)
print("test_data load success")
labels_name = ["name_labels", "dir_labels"]
class_num = {"name_labels": 11, "dir_labels": 4}
bp_name_record_dir = "/media/test/B/python_文件/机器学习大作业/bp_name_record.txt"
bp_dir_record_dir = "/media/test/B/python_文件/机器学习大作业/bp_dir_record.txt"
bp_record_dir = "/media/test/B/python_文件/机器学习大作业/bp_record.txt"
max_acc = 0
max_name_acc = 0
max_name_acc_step = 0


data_placeholder = tf.placeholder(tf.float64, shape=[None,1024])
name_placeholder = tf.placeholder(tf.int64, shape=[None])
dir_placeholder = tf.placeholder(tf.int64, shape=[None])

prediction_dir = BPModule.BP_Dir_Net(data_placeholder,labels_name[1], class_num)
dir_loss = tf.losses.sparse_softmax_cross_entropy(labels=dir_placeholder,
                                                logits=prediction_dir)
dir_predict = tf.argmax(prediction_dir, 1)
dir_correct_prediction = tf.equal(dir_predict, dir_placeholder)
dir_acc = tf.reduce_mean(tf.cast(dir_correct_prediction, tf.float64))
dir_train_op = tf.train.AdamOptimizer(1e-4).minimize(dir_loss)
batch_size = 16
test_batch_size = 8
train_step = 2000
saver = tf.train.Saver(max_to_keep=1)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    for times in range(10): 
        max_acc = 0
        max_dir_acc = 0
        max_dir_acc_step = 0
        f = open(bp_dir_record_dir,'a', encoding="GBK")
        f.write("\n第%d次训练：\n" % (times + 1))
        print("第%d次训练：\n" % (times + 1))
        sess.run(init)
        for i in range(train_step):
            batch_data, batch_dir_labels = train_data.next_batch(
                batch_size, False, False)
            train_dir_acc, train_dir_loss, _ = sess.run([dir_acc, dir_loss, dir_train_op],
                feed_dict={
                    data_placeholder: batch_data,
                    dir_placeholder: batch_dir_labels,
                })
            test_batch_data, test_batch_dir_labels = test_data.next_batch(
                test_batch_size, False, False)
            test_dir_acc = sess.run(dir_acc,
                feed_dict={
                    data_placeholder: test_batch_data,
                    dir_placeholder: test_batch_dir_labels,
                })
            if (i + 1) >= 500 and (i + 1) % 100 == 0:
            # if 1:
                print(
                    '[Train] Step: %d, dir_acc: %4.5f, dir_loss: %4.5f'
                    % (i + 1, train_dir_acc, train_dir_loss))
                print('[Test] Step: %d, dir_acc: %4.5f,' %
                    (i + 1, test_dir_acc))
            if max_dir_acc < test_dir_acc:
                max_dir_acc = test_dir_acc
                max_dir_acc_step = i + 1
        f.write("max_dir_acc: %4.5f, max_dir_acc_step: %d\n" % (max_dir_acc,max_dir_acc_step))
                # saver.save(sess, save_dir + '/mymodel', global_step=i + 1)    
