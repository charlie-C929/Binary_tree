###############################################################
#说明：经整合的初步网络模型DMNv1
#说明：DMN初版
#创建时间：2021/02
#作者：Alison
###############################################################

from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from model7_modify import DMN
import os
import time
#数据所在地址
start = time.time()
component_train_file_path = "G:\Fundamental\Final_Dance\Github_code\Final_Dance\compare\Predict_use.csv"

#利用pandas读入数据
data = pd.read_csv(component_train_file_path)
data_values = data.values


#拆分数据，前9个input wide ，后9个input deep。
x_train_scaled_wide = data_values[:,:9]
x_train_scaled_deep = data_values[:,9:]

filepath = "weights_best.h5"

model = DMN()
model.load_weights(filepath)
predictions = model.predict([x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide,
                x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep,
                x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide,
                x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep,
                x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide,
                x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep,
                x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide,
                x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep,
                x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide,
                x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep,
                x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide,
                x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep,
                x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide,
                x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep,
                x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide,
                x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep])
print(predictions)
end = time.time()
print(end-start)