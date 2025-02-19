# https://blog.csdn.net/zzp20031120/article/details/132296160
# 导入其他需要的库
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import random
import pickle
import glob
from tqdm import tqdm
import os
# 导入语音处理相关的库
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import librosa
from IPython.display import Audio
# 导入所需的模块和类
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Activation, Lambda, Add, Multiply, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.utils import to_categorical, plot_model

## 加载数据集
# 使用glob匹配所有以.trn为扩展名的文件路径
text_paths = glob.glob('data/*.trn')

# 获取匹配到的文件总数
total = len(text_paths)

# 打印总文件数
print(total)

# 使用with语句打开第一个匹配到的文件
with open(text_paths[0], 'r', encoding='utf8') as fr:
    # 读取文件中的所有行并存储在lines列表中
    lines = fr.readlines()

    # 打印读取的行
    print(lines)



######## 提取文本标注和语音文件路径，保留中文并去掉空格 ##########
# 初始化空列表，用于存储处理后的文本和文件路径
texts = []
paths = []

# 遍历匹配到的文件路径
for path in text_paths:
    # 使用with语句打开文件
    with open(path, 'r', encoding='utf8') as fr:
        # 读取文件中的所有行并存储在lines列表中
        lines = fr.readlines()

        # 提取第一行文本并进行处理，去除换行符和空格
        line = lines[0].strip('\n').replace(' ', '')

        # 将处理后的文本添加到texts列表中
        texts.append(line)

        # 将处理后的文件路径添加到paths列表中，去除文件扩展名
        paths.append(path.rstrip('.trn'))

# 打印第一个文件路径和对应的文本内容
print(paths[0], texts[0])

########## 音频数据的加载、处理和可视化
mfcc_dim = 13

def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rms(y=audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr


def visualize(index):
    path = paths[index]
    text = texts[index]
    print('Audio Text:', text)

    audio, sr = load_and_trim(path)
    plt.figure(figsize=(12, 3))
    plt.plot(np.arange(len(audio)), audio)
    plt.title('Raw Audio Signal')
    plt.xlabel('Time')
    plt.ylabel('Audio Amplitude')
    plt.show()

    feature = mfcc(audio, sr, numcep=mfcc_dim, nfft=551)
    print('Shape of MFCC:', feature.shape)

    # Plot MFCC spectrogram with coordinates
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(feature, sr=sr)

    plt.title('Normalized MFCC')
    plt.ylabel('Time')
    plt.xlabel('MFCC Coefficient')
    plt.colorbar(format='%+2.0f dB')
    # Manually set x-axis tick labels for MFCC coefficients
    num_coefficients = feature.shape[0]
    plt.xticks(np.arange(0, 13), np.arange(1, 13 + 1))

    # Manually set y-axis tick labels for time
    num_frames = feature.shape[0]
    print(num_frames)
    time_in_seconds = librosa.frames_to_time(np.arange(0, num_frames, 100), sr=sr)
    time_labels = [t for t in time_in_seconds]
    plt.yticks(np.arange(0, num_frames, 100))

    plt.tight_layout()
    plt.show()

    return path


Audio(visualize(0))




########### 提取音频数据的MFCC特征（大概需要5分钟左右的时间）
features = []

# 使用tqdm来显示循环进度
for i in tqdm(range(total)):
    # 获取当前索引的音频文件路径
    path = paths[i]

    # 加载和修剪音频
    audio, sr = load_and_trim(path)

    # 计算音频的MFCC特征并添加到features列表中
    features.append(mfcc(audio, sr, numcep=mfcc_dim, nfft=551))

# 打印MFCC特征的数量和第一个特征的形状
print(len(features), features[0].shape)

# 从特征列表中随机抽取100个样本
samples = random.sample(features, 100)

# 将样本堆叠成矩阵
samples = np.vstack(samples)

# 计算抽样样本的MFCC均值和标准差
mfcc_mean = np.mean(samples, axis=0)
mfcc_std = np.std(samples, axis=0)
print(mfcc_mean)
print(mfcc_std)

# 对所有特征进行标准化
features = [(feature - mfcc_mean) / (mfcc_std + 1e-14) for feature in features]


######## 建立字典
chars = {}

# 统计所有文本中的字符出现频次
for text in texts:
    for c in text:
        chars[c] = chars.get(c, 0) + 1

# 按字符出现频次排序
chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)

# 仅保留字符列表
chars = [char[0] for char in chars]

# 打印字符数量和前100个字符
print(len(chars), chars[:100])

# 创建字符到ID的映射和ID到字符的映射
char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}

############ 划分训练数据和测试数据
data_index = np.arange(total)
np.random.shuffle(data_index)
train_size = int(0.9 * total)
test_size = total - train_size
train_index = data_index[:train_size]
test_index = data_index[train_size:]

X_train = [features[i] for i in train_index]
Y_train = [texts[i] for i in train_index]
X_test = [features[i] for i in test_index]
Y_test = [texts[i] for i in test_index]


######## 定义批量化生成函数
batch_size = 8


def batch_generator(x, y, batch_size=batch_size):
    offset = 0
    while True:
        offset += batch_size

        if offset == batch_size or offset >= len(x):
            data_index = np.arange(len(x))
            np.random.shuffle(data_index)
            x = [x[i] for i in data_index]
            y = [y[i] for i in data_index]
            offset = batch_size

        X_data = x[offset - batch_size: offset]
        Y_data = y[offset - batch_size: offset]

        X_maxlen = max([X_data[i].shape[0] for i in range(batch_size)])
        Y_maxlen = max([len(Y_data[i]) for i in range(batch_size)])

        X_batch = np.zeros([batch_size, X_maxlen, mfcc_dim])
        Y_batch = np.ones([batch_size, Y_maxlen]) * len(char2id)
        X_length = np.zeros([batch_size, 1], dtype='int32')
        Y_length = np.zeros([batch_size, 1], dtype='int32')

        for i in range(batch_size):
            X_length[i, 0] = X_data[i].shape[0]
            X_batch[i, :X_length[i, 0], :] = X_data[i]

            Y_length[i, 0] = len(Y_data[i])
            Y_batch[i, :Y_length[i, 0]] = [char2id[c] for c in Y_data[i]]

        inputs = {'X': X_batch, 'Y': Y_batch, 'X_length': X_length, 'Y_length': Y_length}
        outputs = {'ctc': np.zeros([batch_size])}

        yield (inputs, outputs)





############### 模型定义
# 定义自定义模块类
class ResidualBlock(Model):
    def __init__(self, filters, kernel_size, dilation_rate):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='causal', activation=None,
                            dilation_rate=dilation_rate)
        self.batchnorm1 = BatchNormalization()
        self.activation_tanh = Activation('tanh')
        self.activation_sigmoid = Activation('sigmoid')
        self.conv2 = Conv1D(filters=filters, kernel_size=1, strides=1, padding='valid', activation=None)
        self.batchnorm2 = BatchNormalization()
        self.add = Add()

    def call(self, inputs):
        hf = self.activation_tanh(self.batchnorm1(self.conv1(inputs)))
        hg = self.activation_sigmoid(self.batchnorm1(self.conv1(inputs)))
        h0 = Multiply()([hf, hg])

        ha = self.activation_tanh(self.batchnorm2(self.conv2(h0)))
        hs = self.activation_tanh(self.batchnorm2(self.conv2(h0)))

        return self.add([ha, inputs]), hs


# 定义其他函数
def conv1d(inputs, filters, kernel_size, dilation_rate):
    return Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='causal', activation=None,
                  dilation_rate=dilation_rate)(inputs)


def batchnorm(inputs):
    return BatchNormalization()(inputs)


def activation(inputs, activation):
    return Activation(activation)(inputs)


# 定义超参数
epochs = 20
num_blocks = 3
filters = 128

# 输入和卷积参数
X = Input(shape=(None, mfcc_dim,), dtype='float32', name='X')
Y = Input(shape=(None,), dtype='float32', name='Y')
X_length = Input(shape=(1,), dtype='int32', name='X_length')
Y_length = Input(shape=(1,), dtype='int32', name='Y_length')

# 构建模型
h0 = activation(batchnorm(conv1d(X, filters, 1, 1)), 'tanh')
shortcut = []
for i in range(num_blocks):
    for r in [1, 2, 4, 8, 16]:
        h0, s = ResidualBlock(filters=filters, kernel_size=7, dilation_rate=r)(h0)
        shortcut.append(s)

h1 = activation(Add()(shortcut), 'relu')
h1 = activation(batchnorm(conv1d(h1, filters, 1, 1)), 'relu')
Y_pred = activation(batchnorm(conv1d(h1, len(char2id) + 1, 1, 1)), 'softmax')
sub_model = Model(inputs=X, outputs=Y_pred)

# 构建整体模型
ctc_loss = Lambda(calc_ctc_loss, output_shape=(1,), name='ctc')([Y, sub_model.output, X_length, Y_length])
model = Model(inputs=[X, Y, X_length, Y_length], outputs=ctc_loss)
optimizer = SGD(learning_rate=0.02, momentum=0.9, nesterov=True, clipnorm=5)
model.compile(loss={'ctc': lambda ctc_true, ctc_pred: ctc_pred}, optimizer=optimizer)

# 回调和训练
checkpointer = ModelCheckpoint(filepath='full_asr.h5', verbose=0)
lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=0.000)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 绘制模型结构图
plot_model(model, to_file='model.png', show_shapes=True, dpi=280)







######## 训练并可视化误差
history = model.fit(
    x=batch_generator(X_train, Y_train),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=batch_generator(X_test, Y_test),
    validation_steps=len(X_test) // batch_size,
    callbacks=[checkpointer, lr_decay, early_stopping])

train_loss = history.history['loss']
valid_loss = history.history['val_loss']
plt.plot(np.linspace(1, epochs, epochs), train_loss, label='train')
plt.plot(np.linspace(1, epochs, epochs), valid_loss, label='valid')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

sub_model.save('sub_asr.h5')

with open('dictionary.pkl', 'wb') as fw:
    pickle.dump([char2id, id2char, mfcc_mean, mfcc_std], fw)


######## 测试模型
from tensorflow.keras.models import load_model
import pickle

with open('dictionary.pkl', 'rb') as fr:
    [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)

sub_model = load_model('sub_asr_1.h5')


def random_predict(x, y):
    index = np.random.randint(len(x))
    feature = x[index]
    text = y[index]

    pred = sub_model.predict(np.expand_dims(feature, axis=0))
    pred_ids = K.eval(K.ctc_decode(pred, [feature.shape[0]], greedy=False, beam_width=10, top_paths=1)[0][0])
    pred_ids = pred_ids.flatten().tolist()

    print('True transcription:\n-- ', text, '\n')
    # 防止音频中出现字典中不存在的字，返回空格代替
    print('Predicted transcription:\n--  ' + ''.join([id2char.get(i, ' ') for i in pred_ids]), '\n')


random_predict(X_train, Y_train)
random_predict(X_test, Y_test)



