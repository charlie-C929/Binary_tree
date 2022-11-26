###############################################################
#说明：未经整合的取得初步结果的模型
#说明：本文件的长处是通过pandas读取csv文件，并经总调试进行训练
#创建时间：2021/02/04
#作者：Alison
###############################################################

from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,Input,Concatenate
from tensorflow.keras.utils import plot_model

component_train_file_path = "G:\Fundamental\Final_Dance\Material_Produce\Final_data.csv"
composite_train_file_path = 'G:\Fundamental\Final_Dance\Material_Produce\\results\\result.csv'

data = pd.read_csv(component_train_file_path)
data_values = data.values

ans = pd.read_csv(composite_train_file_path)
ans_values = ans.values
#print(ans)
# component_raw_train_data = get_dataset(component_train_file_path)
#
# composite_raw_train_data = get_dataset(composite_train_file_path)
#
x_train_all, x_test, y_train_all, y_test = train_test_split(
     data_values, ans_values, random_state = 7)
x_train, x_valid, y_train, y_valid, = train_test_split(
    x_train_all, y_train_all, random_state = 11)

#拆分数据，前9个input wide ，后9个input deep。
x_train_scaled_wide = x_train[:,:9]
x_train_scaled_deep = x_train[:,9:]
x_valid_scaled_wide = x_valid[:,:9]
x_valid_scaled_deep = x_valid[:,9:]
x_test_scaled_wide = x_test[:,:9]
x_test_scaled_deep = x_test[:,9:]

#模型
#Input_Layer
m2_input_layer = Input(shape=[9])
m2_input_layer_2 = Input(shape=[9])
m2_input_layer_3 = Input(shape=[9])
m2_input_layer_4 = Input(shape=[9])
m2_input_layer_5 = Input(shape=[9])
m2_input_layer_6 = Input(shape=[9])
m2_input_layer_7 = Input(shape=[9])
m2_input_layer_8 = Input(shape=[9])
m2_input_layer_9 = Input(shape=[9])
m2_input_layer_10 = Input(shape=[9])
m2_input_layer_11 = Input(shape=[9])
m2_input_layer_12 = Input(shape=[9])
m2_input_layer_13 = Input(shape=[9])
m2_input_layer_14 = Input(shape=[9])
m2_input_layer_15 = Input(shape=[9])
m2_input_layer_16 = Input(shape=[9])
m2_input_layer_17 = Input(shape=[9])
m2_input_layer_18 = Input(shape=[9])
m2_input_layer_19 = Input(shape=[9])
m2_input_layer_20 = Input(shape=[9])
m2_input_layer_21 = Input(shape=[9])
m2_input_layer_22 = Input(shape=[9])
m2_input_layer_23 = Input(shape=[9])
m2_input_layer_24 = Input(shape=[9])
m2_input_layer_25 = Input(shape=[9])
m2_input_layer_26 = Input(shape=[9])
m2_input_layer_27 = Input(shape=[9])
m2_input_layer_28 = Input(shape=[9])
m2_input_layer_29 = Input(shape=[9])
m2_input_layer_30 = Input(shape=[9])
m2_input_layer_31 = Input(shape=[9])
m2_input_layer_32 = Input(shape=[9])
m2_input_layer_33 = Input(shape=[9])
m2_input_layer_34 = Input(shape=[9])
m2_input_layer_35 = Input(shape=[9])
m2_input_layer_36 = Input(shape=[9])
m2_input_layer_37 = Input(shape=[9])
m2_input_layer_38 = Input(shape=[9])
m2_input_layer_39 = Input(shape=[9])
m2_input_layer_40 = Input(shape=[9])
m2_input_layer_41 = Input(shape=[9])
m2_input_layer_42 = Input(shape=[9])
m2_input_layer_43 = Input(shape=[9])
m2_input_layer_44 = Input(shape=[9])
m2_input_layer_45 = Input(shape=[9])
m2_input_layer_46 = Input(shape=[9])
m2_input_layer_47 = Input(shape=[9])
m2_input_layer_48 = Input(shape=[9])
m2_input_layer_49 = Input(shape=[9])
m2_input_layer_50 = Input(shape=[9])
m2_input_layer_51 = Input(shape=[9])
m2_input_layer_52 = Input(shape=[9])
m2_input_layer_53 = Input(shape=[9])
m2_input_layer_54 = Input(shape=[9])
m2_input_layer_55 = Input(shape=[9])
m2_input_layer_56 = Input(shape=[9])
m2_input_layer_57 = Input(shape=[9])
m2_input_layer_58 = Input(shape=[9])
m2_input_layer_59 = Input(shape=[9])
m2_input_layer_60 = Input(shape=[9])
m2_input_layer_61 = Input(shape=[9])
m2_input_layer_62 = Input(shape=[9])
m2_input_layer_63 = Input(shape=[9])
m2_input_layer_64 = Input(shape=[9])

#计算层
m2_dense_layer_1 = Dense(30, activation='relu')(m2_input_layer)
m2_dense_layer_2 = Dense(30, activation='relu')(m2_input_layer_2)
m2_dense_layer_3 = Dense(30, activation='relu')(m2_input_layer_3)
m2_dense_layer_4 = Dense(30, activation='relu')(m2_input_layer_4)
m2_dense_layer_5 = Dense(30, activation='relu')(m2_input_layer_5)
m2_dense_layer_6 = Dense(30, activation='relu')(m2_input_layer_6)
m2_dense_layer_7 = Dense(30, activation='relu')(m2_input_layer_7)
m2_dense_layer_8 = Dense(30, activation='relu')(m2_input_layer_8)
m2_dense_layer_9 = Dense(30, activation='relu')(m2_input_layer_9)
m2_dense_layer_10 = Dense(30, activation='relu')(m2_input_layer_10)
m2_dense_layer_11 = Dense(30, activation='relu')(m2_input_layer_11)
m2_dense_layer_12 = Dense(30, activation='relu')(m2_input_layer_12)
m2_dense_layer_13 = Dense(30, activation='relu')(m2_input_layer_13)
m2_dense_layer_14 = Dense(30, activation='relu')(m2_input_layer_14)
m2_dense_layer_15 = Dense(30, activation='relu')(m2_input_layer_15)
m2_dense_layer_16 = Dense(30, activation='relu')(m2_input_layer_16)
m2_dense_layer_17 = Dense(30, activation='relu')(m2_input_layer_17)
m2_dense_layer_18 = Dense(30, activation='relu')(m2_input_layer_18)
m2_dense_layer_19 = Dense(30, activation='relu')(m2_input_layer_19)
m2_dense_layer_20 = Dense(30, activation='relu')(m2_input_layer_20)
m2_dense_layer_21 = Dense(30, activation='relu')(m2_input_layer_21)
m2_dense_layer_22 = Dense(30, activation='relu')(m2_input_layer_22)
m2_dense_layer_23 = Dense(30, activation='relu')(m2_input_layer_23)
m2_dense_layer_24 = Dense(30, activation='relu')(m2_input_layer_24)
m2_dense_layer_25 = Dense(30, activation='relu')(m2_input_layer_25)
m2_dense_layer_26 = Dense(30, activation='relu')(m2_input_layer_26)
m2_dense_layer_27 = Dense(30, activation='relu')(m2_input_layer_27)
m2_dense_layer_28 = Dense(30, activation='relu')(m2_input_layer_28)
m2_dense_layer_29 = Dense(30, activation='relu')(m2_input_layer_29)
m2_dense_layer_30 = Dense(30, activation='relu')(m2_input_layer_30)
m2_dense_layer_31 = Dense(30, activation='relu')(m2_input_layer_31)
m2_dense_layer_32 = Dense(30, activation='relu')(m2_input_layer_32)
m2_dense_layer_33 = Dense(30, activation='relu')(m2_input_layer_33)
m2_dense_layer_34 = Dense(30, activation='relu')(m2_input_layer_34)
m2_dense_layer_35 = Dense(30, activation='relu')(m2_input_layer_35)
m2_dense_layer_36 = Dense(30, activation='relu')(m2_input_layer_36)
m2_dense_layer_37 = Dense(30, activation='relu')(m2_input_layer_37)
m2_dense_layer_38 = Dense(30, activation='relu')(m2_input_layer_38)
m2_dense_layer_39 = Dense(30, activation='relu')(m2_input_layer_39)
m2_dense_layer_40 = Dense(30, activation='relu')(m2_input_layer_40)
m2_dense_layer_41 = Dense(30, activation='relu')(m2_input_layer_41)
m2_dense_layer_42 = Dense(30, activation='relu')(m2_input_layer_42)
m2_dense_layer_43 = Dense(30, activation='relu')(m2_input_layer_43)
m2_dense_layer_44 = Dense(30, activation='relu')(m2_input_layer_44)
m2_dense_layer_45 = Dense(30, activation='relu')(m2_input_layer_45)
m2_dense_layer_46 = Dense(30, activation='relu')(m2_input_layer_46)
m2_dense_layer_47 = Dense(30, activation='relu')(m2_input_layer_47)
m2_dense_layer_48 = Dense(30, activation='relu')(m2_input_layer_48)
m2_dense_layer_49 = Dense(30, activation='relu')(m2_input_layer_49)
m2_dense_layer_50 = Dense(30, activation='relu')(m2_input_layer_50)
m2_dense_layer_51 = Dense(30, activation='relu')(m2_input_layer_51)
m2_dense_layer_52 = Dense(30, activation='relu')(m2_input_layer_52)
m2_dense_layer_53 = Dense(30, activation='relu')(m2_input_layer_53)
m2_dense_layer_54 = Dense(30, activation='relu')(m2_input_layer_54)
m2_dense_layer_55 = Dense(30, activation='relu')(m2_input_layer_55)
m2_dense_layer_56 = Dense(30, activation='relu')(m2_input_layer_56)
m2_dense_layer_57 = Dense(30, activation='relu')(m2_input_layer_57)
m2_dense_layer_58 = Dense(30, activation='relu')(m2_input_layer_58)
m2_dense_layer_59 = Dense(30, activation='relu')(m2_input_layer_59)
m2_dense_layer_60 = Dense(30, activation='relu')(m2_input_layer_60)
m2_dense_layer_61 = Dense(30, activation='relu')(m2_input_layer_61)
m2_dense_layer_62 = Dense(30, activation='relu')(m2_input_layer_62)
m2_dense_layer_63 = Dense(30, activation='relu')(m2_input_layer_63)
m2_dense_layer_64 = Dense(30, activation='relu')(m2_input_layer_64)


m2_merged_layer_1 = Concatenate()([m2_dense_layer_1, m2_dense_layer_2])
m2_merged_layer_2 = Concatenate()([m2_dense_layer_3, m2_dense_layer_4])
m2_merged_layer_3 = Concatenate()([m2_dense_layer_5, m2_dense_layer_6])
m2_merged_layer_4 = Concatenate()([m2_dense_layer_7, m2_dense_layer_8])
m2_merged_layer_5 = Concatenate()([m2_dense_layer_9, m2_dense_layer_10])
m2_merged_layer_6 = Concatenate()([m2_dense_layer_11, m2_dense_layer_12])
m2_merged_layer_7 = Concatenate()([m2_dense_layer_13, m2_dense_layer_14])
m2_merged_layer_8 = Concatenate()([m2_dense_layer_15, m2_dense_layer_16])
m2_merged_layer_9 = Concatenate()([m2_dense_layer_17, m2_dense_layer_18])
m2_merged_layer_10 = Concatenate()([m2_dense_layer_19, m2_dense_layer_20])
m2_merged_layer_11 = Concatenate()([m2_dense_layer_21, m2_dense_layer_22])
m2_merged_layer_12 = Concatenate()([m2_dense_layer_23, m2_dense_layer_24])
m2_merged_layer_13 = Concatenate()([m2_dense_layer_25, m2_dense_layer_26])
m2_merged_layer_14 = Concatenate()([m2_dense_layer_27, m2_dense_layer_28])
m2_merged_layer_15 = Concatenate()([m2_dense_layer_29, m2_dense_layer_30])
m2_merged_layer_16 = Concatenate()([m2_dense_layer_31, m2_dense_layer_32])
m2_merged_layer_17 = Concatenate()([m2_dense_layer_33, m2_dense_layer_34])
m2_merged_layer_18 = Concatenate()([m2_dense_layer_35, m2_dense_layer_36])
m2_merged_layer_19 = Concatenate()([m2_dense_layer_37, m2_dense_layer_38])
m2_merged_layer_20 = Concatenate()([m2_dense_layer_39, m2_dense_layer_40])
m2_merged_layer_21 = Concatenate()([m2_dense_layer_41, m2_dense_layer_42])
m2_merged_layer_22 = Concatenate()([m2_dense_layer_43, m2_dense_layer_44])
m2_merged_layer_23 = Concatenate()([m2_dense_layer_45, m2_dense_layer_46])
m2_merged_layer_24 = Concatenate()([m2_dense_layer_47, m2_dense_layer_48])
m2_merged_layer_25 = Concatenate()([m2_dense_layer_49, m2_dense_layer_50])
m2_merged_layer_26 = Concatenate()([m2_dense_layer_51, m2_dense_layer_52])
m2_merged_layer_27 = Concatenate()([m2_dense_layer_53, m2_dense_layer_54])
m2_merged_layer_28 = Concatenate()([m2_dense_layer_55, m2_dense_layer_56])
m2_merged_layer_29 = Concatenate()([m2_dense_layer_57, m2_dense_layer_58])
m2_merged_layer_30 = Concatenate()([m2_dense_layer_59, m2_dense_layer_60])
m2_merged_layer_31 = Concatenate()([m2_dense_layer_61, m2_dense_layer_62])
m2_merged_layer_32 = Concatenate()([m2_dense_layer_63, m2_dense_layer_64])

m2_dense_second_layer_1 = Dense(30,activation='relu')(m2_merged_layer_1)
m2_dense_second_layer_2 = Dense(30,activation='relu')(m2_merged_layer_2)
m2_dense_second_layer_3 = Dense(30,activation='relu')(m2_merged_layer_3)
m2_dense_second_layer_4 = Dense(30,activation='relu')(m2_merged_layer_4)
m2_dense_second_layer_5 = Dense(30,activation='relu')(m2_merged_layer_5)
m2_dense_second_layer_6 = Dense(30,activation='relu')(m2_merged_layer_6)
m2_dense_second_layer_7 = Dense(30,activation='relu')(m2_merged_layer_7)
m2_dense_second_layer_8 = Dense(30,activation='relu')(m2_merged_layer_8)
m2_dense_second_layer_9 = Dense(30,activation='relu')(m2_merged_layer_9)
m2_dense_second_layer_10 = Dense(30,activation='relu')(m2_merged_layer_10)
m2_dense_second_layer_11 = Dense(30,activation='relu')(m2_merged_layer_11)
m2_dense_second_layer_12 = Dense(30,activation='relu')(m2_merged_layer_12)
m2_dense_second_layer_13 = Dense(30,activation='relu')(m2_merged_layer_13)
m2_dense_second_layer_14 = Dense(30,activation='relu')(m2_merged_layer_14)
m2_dense_second_layer_15 = Dense(30,activation='relu')(m2_merged_layer_15)
m2_dense_second_layer_16 = Dense(30,activation='relu')(m2_merged_layer_16)
m2_dense_second_layer_17 = Dense(30,activation='relu')(m2_merged_layer_17)
m2_dense_second_layer_18 = Dense(30,activation='relu')(m2_merged_layer_18)
m2_dense_second_layer_19 = Dense(30,activation='relu')(m2_merged_layer_19)
m2_dense_second_layer_20 = Dense(30,activation='relu')(m2_merged_layer_20)
m2_dense_second_layer_21 = Dense(30,activation='relu')(m2_merged_layer_21)
m2_dense_second_layer_22 = Dense(30,activation='relu')(m2_merged_layer_22)
m2_dense_second_layer_23 = Dense(30,activation='relu')(m2_merged_layer_23)
m2_dense_second_layer_24 = Dense(30,activation='relu')(m2_merged_layer_24)
m2_dense_second_layer_25 = Dense(30,activation='relu')(m2_merged_layer_25)
m2_dense_second_layer_26 = Dense(30,activation='relu')(m2_merged_layer_26)
m2_dense_second_layer_27 = Dense(30,activation='relu')(m2_merged_layer_27)
m2_dense_second_layer_28 = Dense(30,activation='relu')(m2_merged_layer_28)
m2_dense_second_layer_29 = Dense(30,activation='relu')(m2_merged_layer_29)
m2_dense_second_layer_30 = Dense(30,activation='relu')(m2_merged_layer_30)
m2_dense_second_layer_31 = Dense(30,activation='relu')(m2_merged_layer_31)
m2_dense_second_layer_32 = Dense(30,activation='relu')(m2_merged_layer_32)

m2_merged_second_layer_1 = Concatenate()([m2_dense_second_layer_1,m2_dense_second_layer_2])
m2_merged_second_layer_2 = Concatenate()([m2_dense_second_layer_3,m2_dense_second_layer_4])
m2_merged_second_layer_3 = Concatenate()([m2_dense_second_layer_5,m2_dense_second_layer_6])
m2_merged_second_layer_4 = Concatenate()([m2_dense_second_layer_7,m2_dense_second_layer_8])
m2_merged_second_layer_5 = Concatenate()([m2_dense_second_layer_9,m2_dense_second_layer_10])
m2_merged_second_layer_6 = Concatenate()([m2_dense_second_layer_11,m2_dense_second_layer_12])
m2_merged_second_layer_7 = Concatenate()([m2_dense_second_layer_13,m2_dense_second_layer_14])
m2_merged_second_layer_8 = Concatenate()([m2_dense_second_layer_15,m2_dense_second_layer_16])
m2_merged_second_layer_9 = Concatenate()([m2_dense_second_layer_17,m2_dense_second_layer_18])
m2_merged_second_layer_10 = Concatenate()([m2_dense_second_layer_19,m2_dense_second_layer_20])
m2_merged_second_layer_11 = Concatenate()([m2_dense_second_layer_21,m2_dense_second_layer_22])
m2_merged_second_layer_12 = Concatenate()([m2_dense_second_layer_23,m2_dense_second_layer_24])
m2_merged_second_layer_13 = Concatenate()([m2_dense_second_layer_25,m2_dense_second_layer_26])
m2_merged_second_layer_14 = Concatenate()([m2_dense_second_layer_27,m2_dense_second_layer_28])
m2_merged_second_layer_15 = Concatenate()([m2_dense_second_layer_29,m2_dense_second_layer_30])
m2_merged_second_layer_16 = Concatenate()([m2_dense_second_layer_31,m2_dense_second_layer_32])

m2_dense_third_layer_1 = Dense(30, activation='relu')(m2_merged_second_layer_1)
m2_dense_third_layer_2 = Dense(30, activation='relu')(m2_merged_second_layer_2)
m2_dense_third_layer_3 = Dense(30, activation='relu')(m2_merged_second_layer_3)
m2_dense_third_layer_4 = Dense(30, activation='relu')(m2_merged_second_layer_4)
m2_dense_third_layer_5 = Dense(30, activation='relu')(m2_merged_second_layer_5)
m2_dense_third_layer_6 = Dense(30, activation='relu')(m2_merged_second_layer_6)
m2_dense_third_layer_7 = Dense(30, activation='relu')(m2_merged_second_layer_7)
m2_dense_third_layer_8 = Dense(30, activation='relu')(m2_merged_second_layer_8)
m2_dense_third_layer_9 = Dense(30, activation='relu')(m2_merged_second_layer_9)
m2_dense_third_layer_10 = Dense(30, activation='relu')(m2_merged_second_layer_10)
m2_dense_third_layer_11 = Dense(30, activation='relu')(m2_merged_second_layer_11)
m2_dense_third_layer_12 = Dense(30, activation='relu')(m2_merged_second_layer_12)
m2_dense_third_layer_13 = Dense(30, activation='relu')(m2_merged_second_layer_13)
m2_dense_third_layer_14 = Dense(30, activation='relu')(m2_merged_second_layer_14)
m2_dense_third_layer_15 = Dense(30, activation='relu')(m2_merged_second_layer_15)
m2_dense_third_layer_16 = Dense(30, activation='relu')(m2_merged_second_layer_16)

m2_merged_third_layer_1 = Concatenate()([m2_dense_third_layer_1,m2_dense_third_layer_2])
m2_merged_third_layer_2 = Concatenate()([m2_dense_third_layer_3,m2_dense_third_layer_4])
m2_merged_third_layer_3 = Concatenate()([m2_dense_third_layer_5,m2_dense_third_layer_6])
m2_merged_third_layer_4 = Concatenate()([m2_dense_third_layer_7,m2_dense_third_layer_8])
m2_merged_third_layer_5 = Concatenate()([m2_dense_third_layer_9,m2_dense_third_layer_10])
m2_merged_third_layer_6 = Concatenate()([m2_dense_third_layer_11,m2_dense_third_layer_12])
m2_merged_third_layer_7 = Concatenate()([m2_dense_third_layer_13,m2_dense_third_layer_14])
m2_merged_third_layer_8 = Concatenate()([m2_dense_third_layer_15,m2_dense_third_layer_16])

m2_dense_fouth_layer_1 = Dense(30,activation='relu')(m2_merged_third_layer_1)
m2_dense_fouth_layer_2 = Dense(30,activation='relu')(m2_merged_third_layer_2)
m2_dense_fouth_layer_3 = Dense(30,activation='relu')(m2_merged_third_layer_3)
m2_dense_fouth_layer_4 = Dense(30,activation='relu')(m2_merged_third_layer_4)
m2_dense_fouth_layer_5 = Dense(30,activation='relu')(m2_merged_third_layer_5)
m2_dense_fouth_layer_6 = Dense(30,activation='relu')(m2_merged_third_layer_6)
m2_dense_fouth_layer_7 = Dense(30,activation='relu')(m2_merged_third_layer_7)
m2_dense_fouth_layer_8 = Dense(30,activation='relu')(m2_merged_third_layer_8)

m2_merged_fouth_layer_1 = Concatenate()([m2_dense_fouth_layer_1,m2_dense_fouth_layer_2])
m2_merged_fouth_layer_2 = Concatenate()([m2_dense_fouth_layer_3,m2_dense_fouth_layer_4])
m2_merged_fouth_layer_3 = Concatenate()([m2_dense_fouth_layer_5,m2_dense_fouth_layer_6])
m2_merged_fouth_layer_4 = Concatenate()([m2_dense_fouth_layer_7,m2_dense_fouth_layer_8])

m2_dense_fifth_layer_1 = Dense(30,activation='relu')(m2_merged_fouth_layer_1)
m2_dense_fifth_layer_2 = Dense(30,activation='relu')(m2_merged_fouth_layer_2)
m2_dense_fifth_layer_3 = Dense(30,activation='relu')(m2_merged_fouth_layer_3)
m2_dense_fifth_layer_4 = Dense(30,activation='relu')(m2_merged_fouth_layer_4)

m2_merge_fifth_layer_1 = Concatenate()([m2_dense_fifth_layer_1,m2_dense_fifth_layer_2])
m2_merge_fifth_layer_2 = Concatenate()([m2_dense_fifth_layer_3,m2_dense_fifth_layer_4])

m2_dense_sixth_layer_1 = Dense(30,activation='relu')(m2_merge_fifth_layer_1)
m2_dense_sixth_layer_2 = Dense(30,activation='relu')(m2_merge_fifth_layer_2)

m2_merge_sixth_layer = Concatenate()([m2_dense_sixth_layer_1,m2_dense_sixth_layer_2])

m2_dense_final_layer = Dense(9,activation='relu')(m2_merge_sixth_layer)





model2 = Model(inputs=[m2_input_layer,m2_input_layer_2,m2_input_layer_3,m2_input_layer_4,m2_input_layer_5,
                       m2_input_layer_6,m2_input_layer_7,m2_input_layer_8,m2_input_layer_9,m2_input_layer_10,
                       m2_input_layer_11,m2_input_layer_12,m2_input_layer_13,m2_input_layer_14,m2_input_layer_15,
                       m2_input_layer_16,m2_input_layer_17,m2_input_layer_18,m2_input_layer_19,m2_input_layer_20,m2_input_layer_21,
                       m2_input_layer_22,m2_input_layer_23,m2_input_layer_24,m2_input_layer_25,m2_input_layer_26,
                       m2_input_layer_27,m2_input_layer_28,m2_input_layer_29,m2_input_layer_30,m2_input_layer_31,
                       m2_input_layer_32,m2_input_layer_33,m2_input_layer_34,m2_input_layer_35,m2_input_layer_36,m2_input_layer_37,
                       m2_input_layer_38,m2_input_layer_39,m2_input_layer_40,m2_input_layer_41,m2_input_layer_42,
                       m2_input_layer_43,m2_input_layer_44,m2_input_layer_45,m2_input_layer_46,m2_input_layer_47,
                       m2_input_layer_48,m2_input_layer_49,m2_input_layer_50,m2_input_layer_51,m2_input_layer_52,m2_input_layer_53,
                       m2_input_layer_54,m2_input_layer_55,m2_input_layer_56,m2_input_layer_57,m2_input_layer_58,
                       m2_input_layer_59,m2_input_layer_60,m2_input_layer_61,m2_input_layer_62,m2_input_layer_63,
                       m2_input_layer_64],
               outputs=m2_dense_final_layer, name="Model_6")
model2.save("model4_initial_weights.h5")
model2.summary()


model2.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])

model2.fit([x_train_scaled_wide,x_train_scaled_deep,x_train_scaled_wide,x_train_scaled_deep,x_train_scaled_wide,x_train_scaled_deep,x_train_scaled_wide,x_train_scaled_deep,
            x_train_scaled_wide,x_train_scaled_deep,x_train_scaled_wide,x_train_scaled_deep,x_train_scaled_wide,x_train_scaled_deep,x_train_scaled_wide,x_train_scaled_deep,
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
            x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep], y_train,
                   validation_data = ([x_valid_scaled_wide,x_valid_scaled_deep,x_valid_scaled_wide,x_valid_scaled_deep,x_valid_scaled_wide,x_valid_scaled_deep,x_valid_scaled_wide,x_valid_scaled_deep,
                                       x_valid_scaled_wide,x_valid_scaled_deep,x_valid_scaled_wide,x_valid_scaled_deep,x_valid_scaled_wide,x_valid_scaled_deep,x_valid_scaled_wide,x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep, x_valid_scaled_wide,
                                       x_valid_scaled_deep, x_valid_scaled_wide, x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep, x_valid_scaled_wide,
                                       x_valid_scaled_deep, x_valid_scaled_wide, x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep, x_valid_scaled_wide,
                                       x_valid_scaled_deep, x_valid_scaled_wide, x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep, x_valid_scaled_wide,
                                       x_valid_scaled_deep, x_valid_scaled_wide, x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep, x_valid_scaled_wide,
                                       x_valid_scaled_deep, x_valid_scaled_wide, x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep, x_valid_scaled_wide,
                                       x_valid_scaled_deep, x_valid_scaled_wide, x_valid_scaled_deep,
                                       x_valid_scaled_wide, x_valid_scaled_deep],y_valid),
                   epochs=10000)

#model2.evaluate(x_test,  y_test, verbose=2)

#打印模型
plot_model(model2, 'model_7.png', show_shapes=True)



