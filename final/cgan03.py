# znxlwm/tensorflow-MNIST-cGAN-cDCGAN
#   https://github.com/znxlwm/tensorflow-MNIST-cGAN-cDCGAN

# 导入必要的库
import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 带泄露的线性整流函数
def leakyRELU(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

# 生成器：G(z)
def generator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()
        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, 128, kernel_initializer=w_init)
        relu1 = tf.nn.relu(dense1)

        dense2 = tf.layers.dense(relu1, 784, kernel_initializer=w_init)
        out = tf.nn.tanh(dense2)
        return out

# 判别器：D(x)
def discriminator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, 128, kernel_initializer=w_init)
        relu1 = leakyRELU(dense1, 0.2)

        dense2 = tf.layers.dense(relu1, 1, kernel_initializer=w_init)
        out = tf.nn.sigmoid(dense2)

        return out, dense2

# 初始化随机噪声和条件变量（图像类别）
onehot = np.eye(10)
temp_z_ = np.random.normal(0, 1, (10, 100))
fixed_z_ = temp_z_
fixed_y_ = np.zeros((10, 1))

for i in range(9):
    fixed_z_ = np.concatenate([fixed_z_, temp_z_], 0)
    temp = np.ones((10,1)) + i
    fixed_y_ = np.concatenate([fixed_y_, temp], 0)

fixed_y_ = onehot[fixed_y_.astype(np.int32)].squeeze()

# 记录中间结果和相关信息
def recordResult(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, y: fixed_y_, isTrain: False})

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()

# 绘制损失函数曲线
def plotLoss(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=1)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# 设置训练参数
BATCH_SIZE = 150
LEARNING_RATE = 0.001
EPOCHS = 200

# 导入 MNIST 数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1
train_label = mnist.train.labels

# 创建输入变量 $x$、随机噪声 $z$、条件变量 $y$
# x: 输入变量
x = tf.placeholder(tf.float32, shape=(None, 784))
# y: 条件变量（图像类别）
y = tf.placeholder(tf.float32, shape=(None, 10))
# z: 随机噪声
z = tf.placeholder(tf.float32, shape=(None, 100))

isTrain = tf.placeholder(dtype=tf.bool)

# 创建生成器
G_z = generator(z, y, isTrain)

# 创建判别器
D_real, D_real_logits = discriminator(x, y, isTrain)
D_fake, D_fake_logits = discriminator(G_z, y, isTrain, reuse=True)

# 设置损失值变量
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([BATCH_SIZE, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([BATCH_SIZE, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([BATCH_SIZE, 1])))

# 设置每层网络的可训练变量
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# 设置每层网络的的优化器
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.5).minimize(G_loss, var_list=G_vars)

# 开启 TensorFlow 的会话并初始化所有变量
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 结果保存路径
root = 'MNIST_cGAN_results/'
model = 'MNIST_cGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

# 中间变量
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []


# 对数据集进行训练
np.random.seed(int(time.time()))
print('+ Training start!')
start_time = time.time()
for epoch in range(EPOCHS):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(len(train_set) // BATCH_SIZE):
        # update discriminator
        x_ = train_set[iter * BATCH_SIZE:(iter + 1) * BATCH_SIZE]
        y_ = train_label[iter * BATCH_SIZE:(iter + 1) * BATCH_SIZE]

        z_ = np.random.normal(0, 1, (BATCH_SIZE, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, y: y_, z: z_, isTrain: True})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (BATCH_SIZE, 100))
        y_ = np.random.randint(0, 10, (BATCH_SIZE, 1))
        y_ = onehot[y_.astype(np.int32)].squeeze()
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, y: y_, isTrain: True})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%3d/%d] ptime: %.2f | d_loss: %.3f | g_loss: %.3f' % ((epoch + 1), EPOCHS, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    recordResult((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('= Avg ptime per epoch: %.3f | total %d epochs time: %.3f' % (np.mean(train_hist['per_epoch_ptimes']), EPOCHS, total_ptime))
print("+ Training finished! Saving training results ...")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

plotLoss(train_hist, save=True, path=root + model + 'train_hist.png')

# png to gif
images = []
for e in range(EPOCHS):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

sess.close()