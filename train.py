import configparser
import os

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model, to_categorical

from generator import Generator
from unet import Unet

#加载配置文件
config = configparser.RawConfigParser()
config.read('config.txt')
#读取对应的参数
experiment_name = config.get('train', 'name')
epochs_num = int(config.get('train', 'epochs_num'))
batch_size = int(config.get('train', 'batch_size'))

#加载数据
datasets = config.get('train', 'datasets')
sub_height = int(config.get('generator', 'sub_height'))
sub_width = int(config.get('generator', 'sub_width'))
x_train, y_train = Generator(datasets, 'train', config)()
#二值化处理  使用了交叉熵损失函数
y_train = to_categorical(y_train)

#CUDA = torch.cuda.is_available()

print(np.max(x_train), np.min(x_train), x_train.shape, x_train.dtype)
print(np.max(y_train), np.min(y_train), y_train.shape, y_train.dtype)

#构建模型并保存
unet = Unet((sub_height, sub_width, 1))
unet.summary()
unet_json = unet.to_json()
open('./logs/' + experiment_name + '_architecture.json', 'w').write(unet_json)
#绘制神经网络结构
plot_model(unet, to_file = './logs/' + experiment_name + '_model.png')

#记录训练过程中最优模型权重文件
checkpointer = ModelCheckpoint(filepath = './logs/'
                                        + experiment_name +'_best_weights.h5',
                                verbose = 1,
                                monitor = 'val_loss',
                                mode = 'auto',
                                save_best_only = True)
#训练
unet.fit(x_train, y_train,
         epochs = epochs_num,
         batch_size = batch_size,
         verbose = 1,
         shuffle = True,
         validation_split = 0.1,
         callbacks = [checkpointer])
#保存
unet.save_weights('./logs/' + experiment_name +'_last_weights.h5',
                   overwrite=True)                   