# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:04:37 2021

@author: Chen PC
"""


import pandas as pd
import numpy as np
import glob
import nibabel
import os
import imageio
import shutil

'''
初始数据转换
'''

def nii_to_img(inputfile,outputfile):
    #获取nii文件名
    data_file = os.listdir(inputfile)
    
    for n_file in data_file:
        img_path = os.path.join(inputfile,n_file)
        #读取nii
        file_data = nibabel.load(img_path)
        
        img_fdata = file_data.get_fdata()
        #去掉nii后缀
        n_file_name = n_file.replace('.nii','')
        
        img_n_file = os.path.join(outputfile,n_file_name)
        #创建nii对应图像的文件夹
        if not os.path.exists(img_n_file):
            #新建文件夹
            os.mkdir(img_n_file)
        #转换图像
        (x,y,z) = file_data.shape
        #Z是图像的序列
        for i in range(x):
            #选择X方向的切片
            silce = img_fdata[i, :, :]
            #保存图片
            imageio.imwrite(os.path.join(img_n_file,'{}.png'.format(i)),silce)

#数据集nii文件路径
datainput = 'input/Training/dataset/'
#数据集图片路径
dataoutput = 'output/img/'
#数据集转换
nii_to_img(datainput,dataoutput)


#标签nii文件路径
labelinput = 'input/Training/label/'
#标签图片路径
labeloutput = 'output/label/'
#标签转换
nii_to_img(labelinput,labeloutput)

