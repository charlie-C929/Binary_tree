###############################################################
#说明：DMNv1的网络部分Depth=7
#说明：基于二叉树网络的神经网络部分
#创建时间：2021/02
#作者：Alison
###############################################################

import os

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,Input,Concatenate
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects
from cosin_dacay import WarmUpCosineDecayScheduler


class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})

#模型
def DMN(x_train_scaled_wide, x_train_scaled_deep,y_train,x_valid_scaled_wide, x_valid_scaled_deep,y_valid,
        tensorboard,checkpoint,checkpoint_path):
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
    m2_dense_layer_1 = Dense(30, activation='Mish')(m2_input_layer)
    m2_dense_layer_2 = Dense(30, activation='Mish')(m2_input_layer_2)
    m2_dense_layer_3 = Dense(30, activation='Mish')(m2_input_layer_3)
    m2_dense_layer_4 = Dense(30, activation='Mish')(m2_input_layer_4)
    m2_dense_layer_5 = Dense(30, activation='Mish')(m2_input_layer_5)
    m2_dense_layer_6 = Dense(30, activation='Mish')(m2_input_layer_6)
    m2_dense_layer_7 = Dense(30, activation='Mish')(m2_input_layer_7)
    m2_dense_layer_8 = Dense(30, activation='Mish')(m2_input_layer_8)
    m2_dense_layer_9 = Dense(30, activation='Mish')(m2_input_layer_9)
    m2_dense_layer_10 = Dense(30, activation='Mish')(m2_input_layer_10)
    m2_dense_layer_11 = Dense(30, activation='Mish')(m2_input_layer_11)
    m2_dense_layer_12 = Dense(30, activation='Mish')(m2_input_layer_12)
    m2_dense_layer_13 = Dense(30, activation='Mish')(m2_input_layer_13)
    m2_dense_layer_14 = Dense(30, activation='Mish')(m2_input_layer_14)
    m2_dense_layer_15 = Dense(30, activation='Mish')(m2_input_layer_15)
    m2_dense_layer_16 = Dense(30, activation='Mish')(m2_input_layer_16)
    m2_dense_layer_17 = Dense(30, activation='Mish')(m2_input_layer_17)
    m2_dense_layer_18 = Dense(30, activation='Mish')(m2_input_layer_18)
    m2_dense_layer_19 = Dense(30, activation='Mish')(m2_input_layer_19)
    m2_dense_layer_20 = Dense(30, activation='Mish')(m2_input_layer_20)
    m2_dense_layer_21 = Dense(30, activation='Mish')(m2_input_layer_21)
    m2_dense_layer_22 = Dense(30, activation='Mish')(m2_input_layer_22)
    m2_dense_layer_23 = Dense(30, activation='Mish')(m2_input_layer_23)
    m2_dense_layer_24 = Dense(30, activation='Mish')(m2_input_layer_24)
    m2_dense_layer_25 = Dense(30, activation='Mish')(m2_input_layer_25)
    m2_dense_layer_26 = Dense(30, activation='Mish')(m2_input_layer_26)
    m2_dense_layer_27 = Dense(30, activation='Mish')(m2_input_layer_27)
    m2_dense_layer_28 = Dense(30, activation='Mish')(m2_input_layer_28)
    m2_dense_layer_29 = Dense(30, activation='Mish')(m2_input_layer_29)
    m2_dense_layer_30 = Dense(30, activation='Mish')(m2_input_layer_30)
    m2_dense_layer_31 = Dense(30, activation='Mish')(m2_input_layer_31)
    m2_dense_layer_32 = Dense(30, activation='Mish')(m2_input_layer_32)
    m2_dense_layer_33 = Dense(30, activation='Mish')(m2_input_layer_33)
    m2_dense_layer_34 = Dense(30, activation='Mish')(m2_input_layer_34)
    m2_dense_layer_35 = Dense(30, activation='Mish')(m2_input_layer_35)
    m2_dense_layer_36 = Dense(30, activation='Mish')(m2_input_layer_36)
    m2_dense_layer_37 = Dense(30, activation='Mish')(m2_input_layer_37)
    m2_dense_layer_38 = Dense(30, activation='Mish')(m2_input_layer_38)
    m2_dense_layer_39 = Dense(30, activation='Mish')(m2_input_layer_39)
    m2_dense_layer_40 = Dense(30, activation='Mish')(m2_input_layer_40)
    m2_dense_layer_41 = Dense(30, activation='Mish')(m2_input_layer_41)
    m2_dense_layer_42 = Dense(30, activation='Mish')(m2_input_layer_42)
    m2_dense_layer_43 = Dense(30, activation='Mish')(m2_input_layer_43)
    m2_dense_layer_44 = Dense(30, activation='Mish')(m2_input_layer_44)
    m2_dense_layer_45 = Dense(30, activation='Mish')(m2_input_layer_45)
    m2_dense_layer_46 = Dense(30, activation='Mish')(m2_input_layer_46)
    m2_dense_layer_47 = Dense(30, activation='Mish')(m2_input_layer_47)
    m2_dense_layer_48 = Dense(30, activation='Mish')(m2_input_layer_48)
    m2_dense_layer_49 = Dense(30, activation='Mish')(m2_input_layer_49)
    m2_dense_layer_50 = Dense(30, activation='Mish')(m2_input_layer_50)
    m2_dense_layer_51 = Dense(30, activation='Mish')(m2_input_layer_51)
    m2_dense_layer_52 = Dense(30, activation='Mish')(m2_input_layer_52)
    m2_dense_layer_53 = Dense(30, activation='Mish')(m2_input_layer_53)
    m2_dense_layer_54 = Dense(30, activation='Mish')(m2_input_layer_54)
    m2_dense_layer_55 = Dense(30, activation='Mish')(m2_input_layer_55)
    m2_dense_layer_56 = Dense(30, activation='Mish')(m2_input_layer_56)
    m2_dense_layer_57 = Dense(30, activation='Mish')(m2_input_layer_57)
    m2_dense_layer_58 = Dense(30, activation='Mish')(m2_input_layer_58)
    m2_dense_layer_59 = Dense(30, activation='Mish')(m2_input_layer_59)
    m2_dense_layer_60 = Dense(30, activation='Mish')(m2_input_layer_60)
    m2_dense_layer_61 = Dense(30, activation='Mish')(m2_input_layer_61)
    m2_dense_layer_62 = Dense(30, activation='Mish')(m2_input_layer_62)
    m2_dense_layer_63 = Dense(30, activation='Mish')(m2_input_layer_63)
    m2_dense_layer_64 = Dense(30, activation='Mish')(m2_input_layer_64)


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

    m2_dense_second_layer_1 = Dense(30,activation='Mish')(m2_merged_layer_1)
    m2_dense_second_layer_2 = Dense(30,activation='Mish')(m2_merged_layer_2)
    m2_dense_second_layer_3 = Dense(30,activation='Mish')(m2_merged_layer_3)
    m2_dense_second_layer_4 = Dense(30,activation='Mish')(m2_merged_layer_4)
    m2_dense_second_layer_5 = Dense(30,activation='Mish')(m2_merged_layer_5)
    m2_dense_second_layer_6 = Dense(30,activation='Mish')(m2_merged_layer_6)
    m2_dense_second_layer_7 = Dense(30,activation='Mish')(m2_merged_layer_7)
    m2_dense_second_layer_8 = Dense(30,activation='Mish')(m2_merged_layer_8)
    m2_dense_second_layer_9 = Dense(30,activation='Mish')(m2_merged_layer_9)
    m2_dense_second_layer_10 = Dense(30,activation='Mish')(m2_merged_layer_10)
    m2_dense_second_layer_11 = Dense(30,activation='Mish')(m2_merged_layer_11)
    m2_dense_second_layer_12 = Dense(30,activation='Mish')(m2_merged_layer_12)
    m2_dense_second_layer_13 = Dense(30,activation='Mish')(m2_merged_layer_13)
    m2_dense_second_layer_14 = Dense(30,activation='Mish')(m2_merged_layer_14)
    m2_dense_second_layer_15 = Dense(30,activation='Mish')(m2_merged_layer_15)
    m2_dense_second_layer_16 = Dense(30,activation='Mish')(m2_merged_layer_16)
    m2_dense_second_layer_17 = Dense(30,activation='Mish')(m2_merged_layer_17)
    m2_dense_second_layer_18 = Dense(30,activation='Mish')(m2_merged_layer_18)
    m2_dense_second_layer_19 = Dense(30,activation='Mish')(m2_merged_layer_19)
    m2_dense_second_layer_20 = Dense(30,activation='Mish')(m2_merged_layer_20)
    m2_dense_second_layer_21 = Dense(30,activation='Mish')(m2_merged_layer_21)
    m2_dense_second_layer_22 = Dense(30,activation='Mish')(m2_merged_layer_22)
    m2_dense_second_layer_23 = Dense(30,activation='Mish')(m2_merged_layer_23)
    m2_dense_second_layer_24 = Dense(30,activation='Mish')(m2_merged_layer_24)
    m2_dense_second_layer_25 = Dense(30,activation='Mish')(m2_merged_layer_25)
    m2_dense_second_layer_26 = Dense(30,activation='Mish')(m2_merged_layer_26)
    m2_dense_second_layer_27 = Dense(30,activation='Mish')(m2_merged_layer_27)
    m2_dense_second_layer_28 = Dense(30,activation='Mish')(m2_merged_layer_28)
    m2_dense_second_layer_29 = Dense(30,activation='Mish')(m2_merged_layer_29)
    m2_dense_second_layer_30 = Dense(30,activation='Mish')(m2_merged_layer_30)
    m2_dense_second_layer_31 = Dense(30,activation='Mish')(m2_merged_layer_31)
    m2_dense_second_layer_32 = Dense(30,activation='Mish')(m2_merged_layer_32)

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

    m2_dense_third_layer_1 = Dense(30, activation='Mish')(m2_merged_second_layer_1)
    m2_dense_third_layer_2 = Dense(30, activation='Mish')(m2_merged_second_layer_2)
    m2_dense_third_layer_3 = Dense(30, activation='Mish')(m2_merged_second_layer_3)
    m2_dense_third_layer_4 = Dense(30, activation='Mish')(m2_merged_second_layer_4)
    m2_dense_third_layer_5 = Dense(30, activation='Mish')(m2_merged_second_layer_5)
    m2_dense_third_layer_6 = Dense(30, activation='Mish')(m2_merged_second_layer_6)
    m2_dense_third_layer_7 = Dense(30, activation='Mish')(m2_merged_second_layer_7)
    m2_dense_third_layer_8 = Dense(30, activation='Mish')(m2_merged_second_layer_8)
    m2_dense_third_layer_9 = Dense(30, activation='Mish')(m2_merged_second_layer_9)
    m2_dense_third_layer_10 = Dense(30, activation='Mish')(m2_merged_second_layer_10)
    m2_dense_third_layer_11 = Dense(30, activation='Mish')(m2_merged_second_layer_11)
    m2_dense_third_layer_12 = Dense(30, activation='Mish')(m2_merged_second_layer_12)
    m2_dense_third_layer_13 = Dense(30, activation='Mish')(m2_merged_second_layer_13)
    m2_dense_third_layer_14 = Dense(30, activation='Mish')(m2_merged_second_layer_14)
    m2_dense_third_layer_15 = Dense(30, activation='Mish')(m2_merged_second_layer_15)
    m2_dense_third_layer_16 = Dense(30, activation='Mish')(m2_merged_second_layer_16)

    m2_merged_third_layer_1 = Concatenate()([m2_dense_third_layer_1,m2_dense_third_layer_2])
    m2_merged_third_layer_2 = Concatenate()([m2_dense_third_layer_3,m2_dense_third_layer_4])
    m2_merged_third_layer_3 = Concatenate()([m2_dense_third_layer_5,m2_dense_third_layer_6])
    m2_merged_third_layer_4 = Concatenate()([m2_dense_third_layer_7,m2_dense_third_layer_8])
    m2_merged_third_layer_5 = Concatenate()([m2_dense_third_layer_9,m2_dense_third_layer_10])
    m2_merged_third_layer_6 = Concatenate()([m2_dense_third_layer_11,m2_dense_third_layer_12])
    m2_merged_third_layer_7 = Concatenate()([m2_dense_third_layer_13,m2_dense_third_layer_14])
    m2_merged_third_layer_8 = Concatenate()([m2_dense_third_layer_15,m2_dense_third_layer_16])

    m2_dense_fouth_layer_1 = Dense(30,activation='Mish')(m2_merged_third_layer_1)
    m2_dense_fouth_layer_2 = Dense(30,activation='Mish')(m2_merged_third_layer_2)
    m2_dense_fouth_layer_3 = Dense(30,activation='Mish')(m2_merged_third_layer_3)
    m2_dense_fouth_layer_4 = Dense(30,activation='Mish')(m2_merged_third_layer_4)
    m2_dense_fouth_layer_5 = Dense(30,activation='Mish')(m2_merged_third_layer_5)
    m2_dense_fouth_layer_6 = Dense(30,activation='Mish')(m2_merged_third_layer_6)
    m2_dense_fouth_layer_7 = Dense(30,activation='Mish')(m2_merged_third_layer_7)
    m2_dense_fouth_layer_8 = Dense(30,activation='Mish')(m2_merged_third_layer_8)

    m2_merged_fouth_layer_1 = Concatenate()([m2_dense_fouth_layer_1,m2_dense_fouth_layer_2])
    m2_merged_fouth_layer_2 = Concatenate()([m2_dense_fouth_layer_3,m2_dense_fouth_layer_4])
    m2_merged_fouth_layer_3 = Concatenate()([m2_dense_fouth_layer_5,m2_dense_fouth_layer_6])
    m2_merged_fouth_layer_4 = Concatenate()([m2_dense_fouth_layer_7,m2_dense_fouth_layer_8])

    m2_dense_fifth_layer_1 = Dense(30,activation='Mish')(m2_merged_fouth_layer_1)
    m2_dense_fifth_layer_2 = Dense(30,activation='Mish')(m2_merged_fouth_layer_2)
    m2_dense_fifth_layer_3 = Dense(30,activation='Mish')(m2_merged_fouth_layer_3)
    m2_dense_fifth_layer_4 = Dense(30,activation='Mish')(m2_merged_fouth_layer_4)

    m2_merge_fifth_layer_1 = Concatenate()([m2_dense_fifth_layer_1,m2_dense_fifth_layer_2])
    m2_merge_fifth_layer_2 = Concatenate()([m2_dense_fifth_layer_3,m2_dense_fifth_layer_4])

    m2_dense_sixth_layer_1 = Dense(30,activation='Mish')(m2_merge_fifth_layer_1)
    m2_dense_sixth_layer_2 = Dense(30,activation='Mish')(m2_merge_fifth_layer_2)

    m2_merge_sixth_layer = Concatenate()([m2_dense_sixth_layer_1,m2_dense_sixth_layer_2])

    m2_dense_final_layer = Dense(9,activation='Mish')(m2_merge_sixth_layer)

    DMN = Model(inputs=[m2_input_layer,m2_input_layer_2,m2_input_layer_3,m2_input_layer_4,m2_input_layer_5,
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
                   outputs=m2_dense_final_layer, name="DMN")

    # #Finetune，只训练最后一层 效果不好
    # for layer in DMN.layers:
    #     if layer.name != 'm2_dense_final_layer':
    #         layer.trainable = False

    #输出模型相关信息
    DMN.summary()

    #模型的更新端设定
    DMN.compile(loss="mean_squared_error",
                optimizer='rmsprop',
                metrics=['accuracy']
                )

    #模型的断点续传部分
    if os.path.exists(checkpoint_path):
        DMN.load_weights(checkpoint_path)
        print("导入模型成功")


    # 余弦退火相关设定
    # 样本总数
    sample_count = 281
    # Total epochs to train.
    epochs = 20000
    # Number of warmup epochs.
    warmup_epoch = 4000
    # Training batch size, set small value here for demonstration purpose.
    batch_size = 32
    # Base learning rate after warmup.这里使用迁移学习，base_rate改为了的十分之一
    learning_rate_base = 0.00001

    total_steps = int(epochs * sample_count / batch_size)

    # Compute the number of warmup batches.
    warmup_steps = int(warmup_epoch * sample_count / batch_size)

    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=4e-06,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=5,
                                            )

    history = DMN.fit([x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep, x_train_scaled_wide,
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
                x_train_scaled_deep, x_train_scaled_wide, x_train_scaled_deep], y_train,
               validation_data=(
               [x_valid_scaled_wide, x_valid_scaled_deep, x_valid_scaled_wide, x_valid_scaled_deep, x_valid_scaled_wide,
                x_valid_scaled_deep, x_valid_scaled_wide, x_valid_scaled_deep,
                x_valid_scaled_wide, x_valid_scaled_deep, x_valid_scaled_wide, x_valid_scaled_deep, x_valid_scaled_wide,
                x_valid_scaled_deep, x_valid_scaled_wide, x_valid_scaled_deep,
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
                x_valid_scaled_wide, x_valid_scaled_deep], y_valid),
               epochs=epochs,
               batch_size=batch_size,
               callbacks=[tensorboard,checkpoint,warm_up_lr])

    return history
