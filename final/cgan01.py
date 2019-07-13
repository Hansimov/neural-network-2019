# GAN[2]：条件CGAN--指定生成结果
#   https://blog.csdn.net/hiudawn/article/details/80752084

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

img_height = 28
img_width = 28
img_size = img_height*img_width
img_cond_size =img_size + 10  # 这里的10是输入拼接了一个10维的one hot向量
batch_size = 128
h1_size = 128
h2_size = 256
max_epoch = 1000000
z_size = 100 # 噪声维度
z_cond_size = z_size + 10  # 加了label的维度
keep_prob = 0.5  # dropout参数
save_path = './condgan_output/'

z = tf.placeholder(tf.float32,shape=[None,z_size])
x = tf.placeholder(tf.float32,shape=[None,img_size])
y = tf.placeholder(tf.float32,shape=[None,10])

def xavier_init(shape):
    '''初始化方法，来源一篇论文，保证每一层都有一致的方差'''
    in_dim = shape[0]
    stddev = 1./tf.sqrt(in_dim/2.)
    return tf.random_normal(shape=shape,stddev=stddev)

def get_z(shape):
    '''生成随机噪声，作为G的输入'''
    return np.random.uniform(-1.,1.,size=shape).astype(np.float32)

def generator(z_prior,y):
    '''生成器，两层感知机，L1用ReLU，Out用sigmoid'''
    # L1
    z_cond = tf.concat([z_prior,y],axis=1)  # 噪声的输入也要加上标签，意思就是“我想伪造谁”

    w1 = tf.Variable(xavier_init([z_cond_size,h1_size]))
    b1 = tf.Variable(tf.zeros([h1_size]),dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(z_cond,w1)+b1)
    # Out
    w2= tf.Variable(xavier_init([h1_size,img_size]))
    b2 = tf.Variable(tf.zeros([img_size]),dtype=tf.float32)
    x_generated = tf.nn.sigmoid(tf.matmul(h1,w2)+b2)
    # 待训练参数要一并返回
    params = [w1,b1,w2,b2]
    return x_generated, params

def discriminator(x,x_generated,keep_prob,y):
    '''判别器，两层感知机，L1用ReLU，Out用sigmoid
     注意判别器用同样的w和b去计算原始样本x和G的生成样本'''
    x_cond = tf.concat([x,y],axis=1)  # 把原始样本和其标签一起放入
    x_generated_cond = tf.concat([x_generated,y],axis=1)  # 把生成样本和其伪造标签一起放入

    # L1
    w1 = tf.Variable(xavier_init([img_cond_size,h1_size]))
    b1 = tf.Variable(tf.zeros([h1_size]),dtype=tf.float32)
    h1_x = tf.nn.dropout(tf.nn.relu(tf.matmul(x_cond,w1)+b1),keep_prob)  # 不加dropout迭代到一定次数会挂掉
    h1_x_generated = tf.nn.dropout(tf.nn.relu(tf.matmul(x_generated_cond,w1)+b1),keep_prob)
    # Out
    w2 = tf.Variable(xavier_init([h1_size,1]))
    b2 = tf.Variable(tf.zeros([1]),dtype=tf.float32)
    d_prob_x = tf.nn.sigmoid(tf.matmul(h1_x,w2)+b2)
    d_prob_x_generated = tf.nn.sigmoid(tf.matmul(h1_x_generated,w2)+b2)

    params = [w1,b1,w2,b2]
    return d_prob_x,d_prob_x_generated,params

def save(samples, index,shape):
    '''只是用来把图片保存到本地，和训练无关'''
    x,y=shape  # 保存图片的宽高（每个单位一张生成数字）
    fig = plt.figure(figsize=(x,y))
    gs = gridspec.GridSpec(x,y)
    gs.update(wspace=0.05,hspace=0.05)

    for i,sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(img_width,img_height),cmap='Greys_r')
    plt.savefig(save_path+'{}.png'.format(str(index).zfill(3)))
    plt.close(fig)

x_generated,g_params = generator(z,y)  # 生产伪造样本
d_prob_real,d_prob_fake,d_params = discriminator(x,x_generated,keep_prob,y)  # 把伪造样本和生成的一并传入计算各自概率

# 这两个是论文里面的那个很长的公式
d_loss = -tf.reduce_mean(tf.log(d_prob_real+1e-30) + tf.log(1.-d_prob_fake+1e-30))  # 不加这个1e-30会出现log(0)
g_loss = -tf.reduce_mean(tf.log(d_prob_fake+1e-30))  # tf有内置的sigmoid_cross_entropy_with_logits可以解决这个问题，但我没用上

g_solver = tf.train.AdamOptimizer(0.001).minimize(g_loss,var_list=g_params)
d_solver = tf.train.AdamOptimizer(0.001).minimize(d_loss,var_list=d_params)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)  # 加载数据集

if not os.path.exists(save_path):
    os.makedirs(save_path)  # 保存图片的位置

for i in range(max_epoch):

    if i % 1000 == 0:  # 这个只是用来保存图片，和训练没什么关系
        labels = [i for i in range(10) for _ in range(10)]  # 我要让他生成的数字，每行相同，每列从0到1递增
        cond_y = sess.run(tf.one_hot(np.array(labels),depth=10))  # 喂的字典不能是tensor，我run成np array
        samples = sess.run(x_generated, feed_dict = {z:get_z([100,z_size]),y:cond_y})
        index = int(i/1000)  # 用来当保存图片的名字
        shape = [10,10]  # 维度和labels的宽高匹配
        save(samples, index, shape)  # 保存图片

    # *主要的训练步骤*
    x_mb,y_mb = mnist.train.next_batch(batch_size)
    _,d_loss_ = sess.run([d_solver,d_loss],feed_dict={x:x_mb,z:get_z([batch_size,z_size]),y:y_mb.astype(np.float32)})
    _,g_loss_ = sess.run([g_solver,g_loss],feed_dict={z:get_z([batch_size,z_size]),y:y_mb.astype(np.float32)})

    if i % 1000 == 0:
        print('iter: %d, d_loss: %.3f, g_loss: %.3f\n' % (i,d_loss_,g_loss_))