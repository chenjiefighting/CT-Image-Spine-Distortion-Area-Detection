

import os
import configparser

import h5py
import numpy as np
from PIL import Image

from utils import load_hdf5, write_hdf5, visualize, group_images


def save_IVDM_to_h5py(train_test, num, config):
    images_path = config.get('IVDM', train_test + '_images_path')
    labels_path = config.get('IVDM', train_test + '_labels_path')
    masks_path = config.get('IVDM', train_test + '_masks_path')
    height = int(config.get('IVDM', 'height'))
    width = int(config.get('IVDM', 'width'))
    #创建给定形状和类型的空数组
    images = np.empty((num, height, width, 1), dtype=np.float32)
    labels = np.empty((num, height, width, 1), dtype=np.float32)
    masks = np.empty((num, height, width, 1), dtype=np.float32)

    files = os.listdir(images_path)
    for i in range(len(files)):
        #读取腰椎图像
        images[i] = np.asarray(Image.open(images_path + files[i])).reshape(
            (height, width, 1))

        #读取标签
        label_name = files[i][0:3] + "_manual.png"
        labels[i] = np.asarray(Image.open(labels_path + label_name)).reshape(
            (height, width, 1))
        
        #读取掩模
        mask_name = ""
        if train_test == "train":
            mask_name = files[i][0:3] + "_training_mask.png"
        elif train_test == "test":
            mask_name = files[i][0:3] + "_test_mask.png"
        masks[i] = np.asarray(Image.open(masks_path + mask_name)).reshape(
            (height, width, 1))

    #打印数据信息
    print('IVDM', train_test)
    print('images', images.shape, images.dtype, np.min(images), np.max(images))
    print('labels', labels.shape, labels.dtype, np.min(labels), np.max(labels))
    print('masks', masks.shape, masks.dtype, np.min(masks), np.max(masks))


    #保存为.hdf5文件
    save_path = config.get('IVDM', 'h5py_save_path')
    if os.path.exists(save_path) == False:
        os.system('mkdir {}'.format(save_path))
    write_hdf5(images, save_path + train_test + '_images' + '.hdf5')
    write_hdf5(labels, save_path + train_test + '_labels' + '.hdf5')
    write_hdf5(masks, save_path + train_test + '_masks' + '.hdf5')





def main():
    #配置文件
    config = configparser.RawConfigParser()
    config.read('config.txt')
    #保存
    save_IVDM_to_h5py('train', 504, config)
    save_IVDM_to_h5py('test', 32, config)


if __name__ == '__main__':
    main()