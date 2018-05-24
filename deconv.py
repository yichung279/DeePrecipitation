from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from glob import glob
from mlxtend.preprocessing import shuffle_arrays_unison
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# for run on only cpu
gpu = False

#global data
#global filename
global batch_count
#data = np.load('data/dataEven_1.npy')
#filename = 'data/dataEven_1.npy'
batch_count = 0

'''
def arr1Dto2D(arr):
    arr2D = [[None] *288] *288
    for i in range(288):
        for j in range(288):
            arr2D[i][j] = arr[288*i + j]
    return arr2D
def get_batch(batch_size):
    
    if 1000 % batch_size != 0:
          print('wrong batch size!')
          os._exit()  

    global data
    global filename
    global batch_count
    
    batch_count += 1
    
    file_size = int(1000 / batch_size) # how much batches in a file
    file_num = int(batch_count / file_size) % 6
    batch_num = int(batch_count % file_size)

    tmp_filename = glob('data/dataEven_*.npy')
    tmp_filename.sort()
    # try to do few times of load
    if tmp_filename[file_num] != filename:
        filename = tmp_filename[file_num]
    
        data = np.load(filename)
        np.random.shuffle(data)


    label = []
    x = []
    label_holder = [0] * 15

    for i in range(batch_num * batch_size, (batch_num+1) * batch_size):
        
        #label
        sublabel = []
        for j in range(data.shape[2]):
            label_holder[data[i][1][j]] = 1
            sublabel.append(label_holder)
            label_holder = [0] * 15
        label.append(sublabel)
        #input
        subX = []
        chanel1 = arr1Dto2D(data[i][1])
        chanel2 = arr1Dto2D(data[i][2])
        chanel3 = arr1Dto2D(data[i][3])
        subX.append(chanel1)
        subX.append(chanel2)
        subX.append(chanel3)
        x.append(np.transpose(subX))
        

    label = np.array(label)
    x = np.array(x)
    return [x, label]

def get_valid_or_test(batch_size, valid = True):
    
    if 1000 % batch_size != 0:
          print('wrong batch size!')
          os._exit()  

    tmp_filename = glob('data/dataOdd_*.npy')
    tmp_filename.sort()
    file_num =np.random.choice(3)
    if valid:
      data = np.load(tmp_filename[file_num])
    else:
      data = np.load(tmp_filename[3 + file_num])
    np.random.shuffle(data)


    label = []
    x = []
    label_holder = [0] * 15

    for i in range(batch_size):
        
        #label
        sublabel = []
        for j in range(data.shape[2]):
            label_holder[data[i][0][j]] = 1
            sublabel.append(label_holder)
            label_holder = [0] * 15
        label.append(sublabel)
        
        #input
        subX = []
        chanel1 = arr1Dto2D(data[i][1])
        chanel2 = arr1Dto2D(data[i][2])
        chanel3 = arr1Dto2D(data[i][3])
        subX.append(chanel1)
        subX.append(chanel2)
        subX.append(chanel3)
        x.append(np.transpose(subX))
        

    label = np.array(label)
    x = np.array(x)
    
    return [x, label]
'''
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name + '_w')

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name + '_b')

def conv2d(x, filter2, padding):
    return tf.nn.conv2d(x, filter2, strides=[1, 1, 1, 1], padding=padding)

def conv_layer(x, w_shape, b_shape, name, padding='SAME'):
    w = weight_variable(w_shape, name)
    b = bias_variable([b_shape], name)
    return tf.nn.relu(conv2d(x, w, padding) + b)

def pool_layer(x):
    if gpu:
        return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1],\
            strides=[1, 2, 2, 1], padding='SAME')
    else:
        return [tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), 0]


def deconv_layer(x, w_shape, b_shape, name, padding='SAME'):
    w = weight_variable(w_shape, name)
    b = bias_variable([b_shape], name)
 
    x_shape = tf.shape(x)
    if padding == 'SAME':
        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], w_shape[2]])
    else:
        out_shape = tf.stack([x_shape[0], w_shape[0], w_shape[1], w_shape[2]])


    node = tf.nn.conv2d_transpose(x, w, out_shape, [1, 1, 1, 1], padding=padding) + b 
    return tf.nn.relu(node)


#!Todo: Here should be trace again
def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    
    return tf.stack(output_list)

def unpool_layer(x, raveled_argmax, out_shape):
    if gpu:
        argmax = unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])
 
        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]
 
        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])
 
        t2 = tf.squeeze(argmax)
        t2 = tf.stack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])
 
        t = tf.concat([t2, t1], 3)
        indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])
 
        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])
 
        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)
    else :
        out = tf.concat([x, x], 3)
        out = tf.concat([out, out], 2)
        return tf.reshape(out, out_shape)

def batch_generator(file_siize):
    if 1000 % file_size != 0:
        print('wrong size!')
        os._exit()

    global batch_count
    file_num = int(batch_count % 11)
    batch_count += 1

    filenames = glob('data/radarTrend.train.*.input.npy') 
    filenames.sort()
    x_train = np.load(filenames[file_num])
    
    filenames = glob('data/radarTrend.train.*.label.npy') 
    filenames.sort()
    y_train = np.load(filenames[file_num])

    x_train, y_train = shuffle_arrays_unison(arrays=[x_train, y_train])
    x_train = np.split(x_train, file_size)
    y_train = np.split(y_train, file_size)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    file_num = np.random.choice(10)

    filenames = glob('data/radarTrend.test.*.input.npy')
    x_valid = np.load(filenames[file_num])

    filenames = glob('data/radarTrend.test.*label.npy')
    y_valid = np.load(filenames[file_num])

    x_valid, y_valid = shuffle_arrays_unison(arrays=[x_valid, y_valid])
    x_valid = np.split(x_valid, file_size)
    y_valid = np.split(y_valid, file_size)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)

    return x_train, y_train, x_valid, y_valid

if __name__ == "__main__":
    #x = tf.placeholder(tf.float32, shape=[None, 288, 288, 3])
    #y_ = tf.placeholder(tf.int64, shape=[None, 288*288, 15])
    x = tf.placeholder(tf.float32, shape=[None, 144, 144, 3])
    y_ = tf.placeholder(tf.int64, shape=[None, 144, 144, 15])
 
    conv_1_1 = conv_layer(x, [3, 3, 3, 64], 64, 'conv1_1')
    conv_1_2 = conv_layer(conv_1_1, [3, 3, 64, 64], 64, 'conv1_2')

    pool1, pool1_argmax = pool_layer(conv_1_2)
   
    conv_2_1 = conv_layer(pool1, [3, 3, 64, 128], 128, 'conv2_1')
    conv_2_2 = conv_layer(conv_2_1, [3, 3, 128, 128], 128, 'conv2_2')
    
    pool2, pool2_argmax = pool_layer(conv_2_2)
    
    conv_3_1 = conv_layer(pool2, [3, 3, 128, 256], 256, 'conv3_1')
    conv_3_2 = conv_layer(conv_3_1, [3, 3, 256, 256] , 256, 'conv3_2')
    conv_3_3 = conv_layer(conv_3_2, [3, 3, 256, 256], 256, 'conv3_3')
    
    pool3, pool3_argmax = pool_layer(conv_3_3)
    
    #conv_4_1 = conv_layer(pool3, [3, 3, 256, 512], 512, 'conv4_1')
    #conv_4_2 = conv_layer(conv_4_1, [3, 3, 512, 512], 512, 'conv4_2')
    #conv_4_3 = conv_layer(conv_4_2, [3, 3, 512, 512], 512, 'conv4_3')

    #pool4, pool4_argmax = pool_layer(conv_4_3)

    #conv_5_1 = conv_layer(pool4, [3, 3, 512, 512], 512, 'conv5_1')
    conv_5_1 = conv_layer(pool3, [3, 3, 256, 512], 512, 'conv5_1')
    conv_5_2 = conv_layer(conv_5_1, [3, 3, 512, 512], 512, 'conv4_2')
    conv_5_3 = conv_layer(conv_5_2, [3, 3, 512, 512], 512, 'conv5_3')

    pool5, pool5_argmax = pool_layer(conv_5_3)

    fc1 = conv_layer(pool5, [9, 9, 512, 4096],4096, 'fc1', padding='VALID')
    fc2 = conv_layer(fc1, [1, 1, 4096, 4096],4096,'fc2', padding='VALID')

    deconv_fc2 = deconv_layer(fc2, [9, 9, 512, 4096], 512, 'fc2_deconv', padding='VALID')

    unpool5 = unpool_layer(deconv_fc2, pool5_argmax, tf.shape(conv_5_3))

    deconv_5_3 = deconv_layer(unpool5, [3, 3, 512, 512], 512, 'deconv5_3')
    deconv_5_2 = deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'conv5_2')
    #deconv_5_1 = deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv5_1')
    deconv_5_1 = deconv_layer(deconv_5_2, [3, 3, 256, 512], 256, 'deconv5_1')

    #unpool4 = unpool_layer(deconv_5_1, pool4_argmax, tf.shape(conv_4_3))

    #deconv_4_3 = deconv_layer(unpool4, [3, 3, 512, 512], 512, 'deconv4_3')
    #deconv_4_2 = deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv4_2')
    #deconv_4_1 = deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv4_1')

    #unpool3 = unpool_layer(deconv_4_1, pool3_argmax, tf.shape(conv_3_3))
    unpool3 = unpool_layer(deconv_5_1, pool3_argmax, tf.shape(conv_3_3))

    deconv_3_3 = deconv_layer(unpool3, [3, 3, 256, 256], 256, 'deconv3_3')
    deconv_3_2 = deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv3_2')
    deconv_3_1 = deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv3_1')

    unpool2 = unpool_layer(deconv_3_1, pool2_argmax, tf.shape(conv_2_2))

    deconv_2_2 = deconv_layer(unpool2, [3, 3, 128, 128], 128, 'deconv2_2')
    deconv_2_1 = deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv2_1')

    unpool1 = unpool_layer(deconv_2_1, pool1_argmax, tf.shape(conv_1_2))

    deconv_1_2 = deconv_layer(unpool1, [3, 3, 64, 64], 64, 'deconv1_2')
    deconv_1_1 = deconv_layer(deconv_1_2, [3, 3, 64, 64], 64, 'deconv1_1')

    score = deconv_layer(deconv_1_1, [1, 1, 15, 64], 15, 'score')#0(grayscale)~15

    labels = tf.cast(tf.reshape(y_, (-1, 15)), tf.float32)
    logits = tf.cast(tf.reshape(score, (-1, 15)), tf.float32)
    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-6).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(score, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
     
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 25 # 40 => max: 10.5G, avg: 9G
        epoches =100
        for e in range(epoches):
            for j in range(11):
                file_size =int(1000 / batch_size)
                x_train, y_train, x_valid, y_valid = batch_generator(file_size)
                for i in range(file_size):
                    train_step.run(feed_dict={x: x_train[i], y_: y_train[i]})

                    train_accuracy = accuracy.eval(feed_dict={x: x_train[i], y_: y_train[i]}) 
                    train_loss = loss.eval(feed_dict={x: x_train[i], y_: y_train[i]})
         
                    valid_accuracy = accuracy.eval(feed_dict={x: x_valid[i], y_: y_valid[i]})
                    valid_loss = loss.eval(feed_dict={x: x_valid[i], y_: y_valid[i]})

                    print('epochs: %d,step %d: ' % ((e + 1), (j*file_size + i + 1)))
                    print('    |--->training accuracy %g' % (train_accuracy))
                    print('    |--->training loss %g' % (train_loss))
                    print('    |--->valid accuract %g' % (valid_accuracy))
                    print('    +--->valid loss %g' % (valid_loss))
             
                    if i % 10 == 0: 
                       save_path = saver.save(sess, "./model/model%g.ckpt/"%(i))
