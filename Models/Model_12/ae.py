import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

df_test = pd.read_pickle('./all_jets_test_4D_100_percent.pkl')
df_train = pd.read_pickle('./all_jets_train_4D_100_percent.pkl')

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

df_train.reset_index(drop=True, inplace=True)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

train_samples = df_train.shape[0]
val_samples = int(train_samples * 0.2)
test_samples = df_test.shape[0]

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

val_indices = np.random.randint(train_samples, size=val_samples)

df_val = df_train.iloc[val_indices, :]
df_val.reset_index(drop=True, inplace=True)

df_train = df_train.drop(df_train.index[val_indices])
df_train.reset_index(drop=True, inplace=True)

train_mean = df_train.mean()
train_std = df_train.std()

print('Train_mean: ', train_mean)
print('Train_std: ', train_std)

df_train = (df_train - train_mean) / train_std
df_test = (df_test - train_mean) / train_std
df_val = (df_val - train_mean) / train_std

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

LEARNING_RATE = 0.001
BATCH_SIZE = 150
EPOCHS = 500
BATCHES = df_train.shape[0] // BATCH_SIZE
BETA = 0.01
EPSILON = 1e-3

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

with tf.variable_scope('Placeholder'):
    X = tf.placeholder(tf.float32, shape=(None, 4), name='Input')
    # keep_prob = tf.placeholder(tf.float32, name='Keep_Probability')
    is_training = tf.placeholder_with_default(False, (), name='Batch_Flag')
        
with tf.variable_scope('Encoder'):
    kernel_initializer = tf.glorot_uniform_initializer()
    bias_initializer = tf.glorot_normal_initializer()

    encoder_1 = tf.layers.dense(X, 50, name='encoder_1', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    encoder_1_bn = tf.layers.batch_normalization(encoder_1, training=is_training, momentum=0.9, name='encoder_1_bn')
    encoder_1_ac = tf.nn.tanh(encoder_1_bn, name='encoder_1_ac')

    encoder_2 = tf.layers.dense(encoder_1_ac, 30, name='encoder_2', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    encoder_2_bn = tf.layers.batch_normalization(encoder_2, training=is_training, momentum=0.9)
    encoder_2_ac = tf.nn.tanh(encoder_2_bn)

    encoder_3 = tf.layers.dense(encoder_2_ac, 20, name='encoder_3', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    encoder_3_bn = tf.layers.batch_normalization(encoder_3, training=is_training, momentum=0.9)
    encoder_3_ac = tf.nn.tanh(encoder_3_bn)

    encoder_4 = tf.layers.dense(encoder_3_ac, 3, name='encoder_4', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    encoder_4_bn = tf.layers.batch_normalization(encoder_4, training=is_training, momentum=0.9)
    encoder_4_ac = tf.nn.tanh(encoder_4_bn)

    # encoder_5 = tf.layers.dense(encoder_4_ac, 30, name='encoder_5', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    # # encoder_5_bn = tf.layers.batch_normalization(encoder_5, training=is_training, momentum=0.9)
    # encoder_5_ac = tf.nn.tanh(encoder_5)

    # encoder_6 = tf.layers.dense(encoder_5_ac, 3, name='encoder_6', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    # # encoder_6_bn = tf.layers.batch_normalization(encoder_6, training=is_training, momentum=0.9)
    # encoder_6_ac = tf.nn.tanh(encoder_6)

with tf.variable_scope('Decoder'):

    decoder_1 = tf.layers.dense(encoder_4_ac, 20, name='decoder_1', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    decoder_1_bn = tf.layers.batch_normalization(decoder_1, training=is_training, momentum=0.9)
    decoder_1_ac = tf.nn.tanh(decoder_1_bn)

    decoder_2 = tf.layers.dense(decoder_1_ac, 20, name='decoder_2', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    decoder_2_bn = tf.layers.batch_normalization(decoder_2, training=is_training, momentum=0.9)
    decoder_2_ac = tf.nn.tanh(decoder_2_bn)

    decoder_3 = tf.layers.dense(decoder_2_ac, 20, name='decoder_3', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    decoder_3_bn = tf.layers.batch_normalization(decoder_3, training=is_training, momentum=0.9)
    decoder_3_ac = tf.nn.tanh(decoder_3_bn)

    decoder_4 = tf.layers.dense(decoder_3_ac, 4, name='decoder_4', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    # decoder_4_bn = tf.layers.batch_normalization(decoder_4, training=is_training, momentum=0.9)
    # decoder_4_ac = tf.nn.tanh(decoder_4)

    # decoder_5 = tf.layers.dense(decoder_4_ac, 500, name='decoder_5', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    # # decoder_5_bn = tf.layers.batch_normalization(decoder_5, training=is_training, momentum=0.9)
    # decoder_5_ac = tf.nn.tanh(decoder_5)

    # decoder_6 = tf.layers.dense(decoder_5_ac, 4, name='decoder_6', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
    # # decoder_6_bn = tf.layers.batch_normalization(decoder_6, training=is_training, momentum=0.9)
    # # decoder_6_ac = tf.nn.elu(decoder_6_bn)

with tf.variable_scope('Loss'):
    mse_loss = tf.reduce_mean(tf.squared_difference(X, decoder_4))
    # reg_loss = BETA * (W1_reg + W2_reg + W3_reg + W4_reg + W5_reg + W6_reg + W7_reg + W8_reg)
    loss = mse_loss

with tf.variable_scope('Train'):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.variable_scope('Layer_name'):
    weight_e1 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_1' + '/kernel:0')
    weight_e2 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_2' + '/kernel:0')
    weight_e3 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_3' + '/kernel:0')
    weight_e4 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_4' + '/kernel:0')
    # weight_e5 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_5' + '/kernel:0')
    # weight_e6 = tf.get_default_graph().get_tensor_by_name('Encoder/' + 'encoder_6' + '/kernel:0')
    
    weight_d1 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_1' + '/kernel:0')
    weight_d2 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_2' + '/kernel:0')
    weight_d3 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_3' + '/kernel:0')
    weight_d4 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_4' + '/kernel:0')
    # weight_d5 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_5' + '/kernel:0')
    # weight_d6 = tf.get_default_graph().get_tensor_by_name('Decoder/' + 'decoder_6' + '/kernel:0')

with tf.variable_scope('Summary'):
    # tf.summary.scalar('MSE_Loss', mse_loss)
    tf.summary.scalar('Total_Loss', loss)
    # tf.summary.scalar('Penalty', reg_loss)
    tf.summary.histogram('E1', weight_e1)
    tf.summary.histogram('E2', weight_e2)
    tf.summary.histogram('E3', weight_e3)
    tf.summary.histogram('E4', weight_e4)
    # tf.summary.histogram('E5', weight_e5)
    # tf.summary.histogram('E6', weight_e6)
    tf.summary.histogram('D1', weight_d1)
    tf.summary.histogram('D2', weight_d2)
    tf.summary.histogram('D3', weight_d3)
    tf.summary.histogram('D4', weight_d4)
    # tf.summary.histogram('D5', weight_d5)
    # tf.summary.histogram('D6', weight_d6)

merged_summary = tf.summary.merge_all()
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    writer = tf.summary.FileWriter('Tensorflow/LOGS/')
    writer.add_graph(sess.graph)
    saver = tf.train.Saver(max_to_keep=10)
    train_loss, validation_loss, point = [], [], []

    for epoch in range(1, EPOCHS+1):
        training_loss_ = 0
        for batch in range(BATCHES):
            indices = np.random.randint(df_train.shape[0], size=BATCH_SIZE)
            x_main_tr = df_train.iloc[indices, :]

            _, _, tr_l, s = sess.run([train_step, extra_update_ops, loss, merged_summary], feed_dict={X: x_main_tr, is_training: True})
            training_loss_ += tr_l
            writer.add_summary(s, epoch)

        df_val = df_val.sample(frac=1).reset_index(drop=True)
        val_l = loss.eval(feed_dict={X: df_val})
        p = decoder_4.eval(feed_dict={X: df_train})
        print('Epoch: {0}/{1}      Loss: {2}      Val_loss: {3}'.format(epoch, EPOCHS, training_loss_, val_l))
        train_loss.append(training_loss_)
        validation_loss.append(val_l)
        point.append(p[0, :])
        if epoch % 5 == 0:
            saver.save(sess, 'Tensorflow/Checkpoints/epoch_{}'.format(epoch), )
    output_after_training = decoder_4.eval(feed_dict={X: df_test})

point, train_loss, validation_loss = np.array(point), np.array(train_loss), np.array(validation_loss)

np.save('./Training_loss.npy', train_loss)
np.save('./Validation_loss.npy', validation_loss)
np.save('./Data_point.npy', point)

output_after_training = pd.DataFrame(output_after_training, columns=df_test.columns)
output_after_training.to_csv('VS_CODE_Output_after_training.csv', encoding='utf-8')
df_test.to_csv('VS_CODE_Test_Set.csv', encoding='utf-8')
residual_errors = (df_test - output_after_training) / df_test
print(residual_errors.mean())


## AE_3D_50_cone with BN
