# -*- coding: utf-8 -*-


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
import tensorflow as tf
from networks import NetworkAlbertTextCNN
from classifier_utils import get_features
from hyperparameters import Hyperparamters as hp
from utils import select, time_now_string

pwd = os.path.dirname(os.path.abspath(__file__))
MODEL = NetworkAlbertTextCNN(is_training=True)  # 初始化模型

label_balance = np.array(
    [1.0, 0.8, 0.8520547945205479, 0.8712785388127854, 0.88324200913242, 0.9024200913242009, 0.9082191780821918,
     0.9171689497716895, 0.9176255707762557, 0.9193150684931507, 0.9197260273972603, 0.9265753424657535,
     0.9350228310502283, 0.9364840182648402, 0.9401826484018265, 0.9425570776255707, 0.9430593607305936,
     0.944337899543379, 0.9504109589041095, 0.9525570776255707, 0.9634703196347032, 0.9659360730593607,
     0.9688127853881279, 0.968904109589041, 0.9702283105022831, 0.9716438356164383, 0.9719634703196347,
     0.9721917808219178, 0.9743835616438357, 0.9750684931506849, 0.9758447488584475, 0.9768036529680365,
     0.9780365296803653, 0.9813698630136987, 0.9826940639269406, 0.9830593607305936, 0.9835159817351599,
     0.9838356164383562, 0.9859817351598174, 0.9860730593607306, 0.9860730593607306, 0.9865753424657534,
     0.9867579908675799, 0.9884931506849315, 0.9889041095890411, 0.9906849315068493, 0.9909132420091324,
     0.9923744292237443, 0.9933789954337899, 0.9937442922374429, 0.9941552511415526, 0.994703196347032,
     0.9952511415525114, 0.9955251141552511, 0.9962557077625571, 0.9974885844748859, 0.9982191780821917,
     0.998310502283105, 0.9988584474885844, 0.9998173515981735, 1.0])

# Get data features
input_ids, input_masks, segment_ids, label_ids = get_features()
label_tmp = []
for label_one in label_ids:
    label_one = label_one.astype(np.float)
    label_tmp.append(label_one * label_balance)  # 对标签进行加权
label_ids = label_tmp
num_train_samples = len(input_ids)
indexs = np.arange(num_train_samples)  # 固定步长的排列
num_batchs = int((num_train_samples - 1) / hp.batch_size) + 1
print('Number of batch:', num_batchs)

# Set up the graph
saver = tf.train.Saver(max_to_keep=hp.max_to_keep)  # 最多保存模型的数量
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Load model saved before
MODEL_SAVE_PATH = os.path.join(pwd, hp.file_save_model)
ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Restored model!')

with sess.as_default():
    # Tensorboard writer
    writer = tf.summary.FileWriter(hp.logdir, sess.graph)  # 指定一个文件用来保存图
    for i in range(hp.num_train_epochs):
        np.random.shuffle(indexs)  # 重新排序返回一个随机序列
        for j in range(num_batchs - 1):
            # Get ids selected
            i1 = indexs[j * hp.batch_size:min((j + 1) * hp.batch_size, num_train_samples)]

            # Get features
            input_id_ = select(input_ids, i1)
            input_mask_ = select(input_masks, i1)
            segment_id_ = select(segment_ids, i1)
            label_id_ = select(label_ids, i1)

            # Feed dict
            fd = {MODEL.input_ids: input_id_,
                  MODEL.input_masks: input_mask_,
                  MODEL.segment_ids: segment_id_,
                  MODEL.label_ids: label_id_}

            # Optimizer
            sess.run(MODEL.optimizer, feed_dict=fd)

            # Tensorboard
            if j % hp.summary_step == 0:
                summary, glolal_step = sess.run([MODEL.merged, MODEL.global_step], feed_dict=fd)
                writer.add_summary(summary, glolal_step)

            # Save Model
            if j % (num_batchs // hp.num_saved_per_epoch) == 0:
                if not os.path.exists(os.path.join(pwd, hp.file_save_model)):
                    os.makedirs(os.path.join(pwd, hp.file_save_model))
                saver.save(sess, os.path.join(pwd, hp.file_save_model, 'model' + '_%s_%s.ckpt' % (str(i), str(j))))

            # Log
            if j % hp.print_step == 0:
                fd = {MODEL.input_ids: input_id_,
                      MODEL.input_masks: input_mask_,
                      MODEL.segment_ids: segment_id_,
                      MODEL.label_ids: label_id_}
                loss = sess.run(MODEL.loss, feed_dict=fd)
                accuracy = sess.run(MODEL.accuracy, feed_dict=fd)
                f1 = sess.run(MODEL.f1, feed_dict=fd)
                precision = sess.run(MODEL.precision, feed_dict=fd)
                recall = sess.run(MODEL.recall, feed_dict=fd)
                print('Time:%s, Epoch:%s, Batch number:%s/%s, Loss:%s, Accuracy:%s, F1:%s, Precision:%s, Recall%s' % (
                    time_now_string(), str(i), str(j), str(num_batchs), str(loss), str(accuracy), str(f1),
                    str(precision), str(recall)))
    print('Train finished')
