###############################################################
#说明：经整合的初步网络模型DMNv1
#说明：DMN初版
#创建时间：2021/02
#作者：Alison
###############################################################

from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,Input,Concatenate
from tensorflow.keras.utils import plot_model
from model_finetune import DMN
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # 数据所在地址
    # component_train_file_path = "G:\Fundamental\Final_Dance\Material_Produce\Final_data.csv"
    # composite_train_file_path = 'G:\Fundamental\Final_Dance\Material_Produce\\results\\result.csv'
    component_train_file_path = "G:\Fundamental\Final_Dance\Material_Produce\Random_data.csv"
    composite_train_file_path = 'G:\Fundamental\Final_Dance\Material_Produce\\random_results\\result.csv'
    #利用pandas读入数据
    data = pd.read_csv(component_train_file_path)
    data_values = data.values

    ans = pd.read_csv(composite_train_file_path)
    ans_values = ans.values

    # 利用sklearn划分
    x_train_all, x_test, y_train_all, y_test = train_test_split(
         data_values, ans_values, random_state = 7)
    x_train, x_valid, y_train, y_valid, = train_test_split(
        x_train_all, y_train_all, random_state = 11)

    # 拆分数据，前9个input wide ，后9个input deep。
    x_train_scaled_wide = x_train[:,:9]
    x_train_scaled_deep = x_train[:,9:]
    x_valid_scaled_wide = x_valid[:,:9]
    x_valid_scaled_deep = x_valid[:,9:]
    x_test_scaled_wide = x_test[:,:9]
    x_test_scaled_deep = x_test[:,9:]

    #训练可视化,从这里可以导出数据，完成
    log_dir= os.path.join('logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tensorboard = TensorBoard(log_dir=log_dir)

    # 模型保存&断点续传
    # 保存地址
    filepath = ""
    # 还原点选项：保存地址、监视器为验证集的准确率，啰嗦=1，详细显示过程，只保存最好的权重，mode就是最大值
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')


    history = DMN(x_train_scaled_wide,x_train_scaled_deep,y_train,x_valid_scaled_wide,x_valid_scaled_deep,y_valid,
                  tensorboard,checkpoint,filepath)





    # #  使用history将训练集和测试集的loss和acc调出来
    # acc = history.history['accuracy']  # 训练集准确率
    # val_acc = history.history['val_accuracy']  # 测试集准确率
    # loss = history.history['loss']  # 训练集损失
    # val_loss = history.history['val_loss']  # 测试集损失
    # #  打印acc和loss，采用一个图进行显示。
    # #  将acc打印出来。
    # plt.subplot(1, 2, 1)  # 将图像分为一行两列，将其显示在第一列
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)  # 将其显示在第二列
    # plt.plot(loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.show()
