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
prediction_name = BPModule.BP_Name_Net(data_placeholder, labels_name[0], class_num)
name_loss = tf.losses.sparse_softmax_cross_entropy(labels=name_placeholder,
                                                logits=prediction_name)
name_predict = tf.argmax(prediction_name, 1)
name_correct_prediction = tf.equal(name_predict, name_placeholder)
name_acc = tf.reduce_mean(tf.cast(name_correct_prediction, tf.float64))
name_train_op = tf.train.AdamOptimizer(1e-4).minimize(name_loss)

batch_size = 16
test_batch_size = 8
train_step = 3000
saver = tf.train.Saver(max_to_keep=1)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    for times in range(10):
        max_acc = 0
        max_name_acc = 0
        max_name_acc_step = 0
        f = open(bp_name_record_dir,'a', encoding="GBK")
        f.write("\n第%d次训练：\n" % (times + 1))
        print("第%d次训练：\n" % (times + 1))
        sess.run(init)
        for i in range(train_step):
            batch_data, batch_name_labels = train_data.next_batch(
                batch_size, False, True)
            train_name_acc, train_name_loss, _ = sess.run([name_acc, name_loss, name_train_op],
                feed_dict={
                    data_placeholder: batch_data,
                    name_placeholder: batch_name_labels,
                })
            test_batch_data, test_batch_name_labels = test_data.next_batch(
                test_batch_size, False, True)
            test_name_acc = sess.run(name_acc,
                feed_dict={
                    data_placeholder: test_batch_data,
                    name_placeholder: test_batch_name_labels,
                })
            # if (i + 1) >= 50 and (i + 1) % 10 == 0:
            if 1:
                print(
                    '[Train] Step: %d, name_acc: %4.5f, name_loss: %4.5f'
                    % (i + 1, train_name_acc, train_name_loss))
                print('[Test] Step: %d, name_acc: %4.5f,' %
                    (i + 1, test_name_acc))
            if max_name_acc < test_name_acc:
                max_name_acc = test_name_acc
                max_name_acc_step = i + 1
        f.write("max_name_acc: %4.5f, max_name_acc_step: %d\n" %(
                     max_name_acc, max_name_acc_step))
                # saver.save(sess, save_dir + '/mymodel', global_step=i + 1)
