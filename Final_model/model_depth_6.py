###############################################################
#说明：DMNv1的网络部分Depth=6
#说明：基于二叉树网络的神经网络部分
#创建时间：2021/03
#作者：Alison
###############################################################
import os

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense,Input,Concatenate
import tensorflow.keras as keras
from cosin_dacay import WarmUpCosineDecayScheduler

#模型
def DMN(x_train_scaled_wide, x_train_scaled_deep,y_train,x_valid_scaled_wide, x_valid_scaled_deep,y_valid,
        tensorboard,checkpoint,checkpoint_path):
    # 模型
    # Input_Layer
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

    # 计算层
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

    m2_dense_second_layer_1 = Dense(30, activation='relu')(m2_merged_layer_1)
    m2_dense_second_layer_2 = Dense(30, activation='relu')(m2_merged_layer_2)
    m2_dense_second_layer_3 = Dense(30, activation='relu')(m2_merged_layer_3)
    m2_dense_second_layer_4 = Dense(30, activation='relu')(m2_merged_layer_4)
    m2_dense_second_layer_5 = Dense(30, activation='relu')(m2_merged_layer_5)
    m2_dense_second_layer_6 = Dense(30, activation='relu')(m2_merged_layer_6)
    m2_dense_second_layer_7 = Dense(30, activation='relu')(m2_merged_layer_7)
    m2_dense_second_layer_8 = Dense(30, activation='relu')(m2_merged_layer_8)
    m2_dense_second_layer_9 = Dense(30, activation='relu')(m2_merged_layer_9)
    m2_dense_second_layer_10 = Dense(30, activation='relu')(m2_merged_layer_10)
    m2_dense_second_layer_11 = Dense(30, activation='relu')(m2_merged_layer_11)
    m2_dense_second_layer_12 = Dense(30, activation='relu')(m2_merged_layer_12)
    m2_dense_second_layer_13 = Dense(30, activation='relu')(m2_merged_layer_13)
    m2_dense_second_layer_14 = Dense(30, activation='relu')(m2_merged_layer_14)
    m2_dense_second_layer_15 = Dense(30, activation='relu')(m2_merged_layer_15)
    m2_dense_second_layer_16 = Dense(30, activation='relu')(m2_merged_layer_16)

    m2_merged_second_layer_1 = Concatenate()([m2_dense_second_layer_1, m2_dense_second_layer_2])
    m2_merged_second_layer_2 = Concatenate()([m2_dense_second_layer_3, m2_dense_second_layer_4])
    m2_merged_second_layer_3 = Concatenate()([m2_dense_second_layer_5, m2_dense_second_layer_6])
    m2_merged_second_layer_4 = Concatenate()([m2_dense_second_layer_7, m2_dense_second_layer_8])
    m2_merged_second_layer_5 = Concatenate()([m2_dense_second_layer_9, m2_dense_second_layer_10])
    m2_merged_second_layer_6 = Concatenate()([m2_dense_second_layer_11, m2_dense_second_layer_12])
    m2_merged_second_layer_7 = Concatenate()([m2_dense_second_layer_13, m2_dense_second_layer_14])
    m2_merged_second_layer_8 = Concatenate()([m2_dense_second_layer_15, m2_dense_second_layer_16])

    m2_dense_third_layer_1 = Dense(30, activation='relu')(m2_merged_second_layer_1)
    m2_dense_third_layer_2 = Dense(30, activation='relu')(m2_merged_second_layer_2)
    m2_dense_third_layer_3 = Dense(30, activation='relu')(m2_merged_second_layer_3)
    m2_dense_third_layer_4 = Dense(30, activation='relu')(m2_merged_second_layer_4)
    m2_dense_third_layer_5 = Dense(30, activation='relu')(m2_merged_second_layer_5)
    m2_dense_third_layer_6 = Dense(30, activation='relu')(m2_merged_second_layer_6)
    m2_dense_third_layer_7 = Dense(30, activation='relu')(m2_merged_second_layer_7)
    m2_dense_third_layer_8 = Dense(30, activation='relu')(m2_merged_second_layer_8)

    m2_merged_third_layer_1 = Concatenate()([m2_dense_third_layer_1, m2_dense_third_layer_2])
    m2_merged_third_layer_2 = Concatenate()([m2_dense_third_layer_3, m2_dense_third_layer_4])
    m2_merged_third_layer_3 = Concatenate()([m2_dense_third_layer_5, m2_dense_third_layer_6])
    m2_merged_third_layer_4 = Concatenate()([m2_dense_third_layer_7, m2_dense_third_layer_8])

    m2_dense_fouth_layer_1 = Dense(30, activation='relu')(m2_merged_third_layer_1)
    m2_dense_fouth_layer_2 = Dense(30, activation='relu')(m2_merged_third_layer_2)
    m2_dense_fouth_layer_3 = Dense(30, activation='relu')(m2_merged_third_layer_3)
    m2_dense_fouth_layer_4 = Dense(30, activation='relu')(m2_merged_third_layer_4)

    m2_merged_fouth_layer_1 = Concatenate()([m2_dense_fouth_layer_1, m2_dense_fouth_layer_2])
    m2_merged_fouth_layer_2 = Concatenate()([m2_dense_fouth_layer_3, m2_dense_fouth_layer_4])

    m2_dense_fifth_layer_1 = Dense(30, activation='relu')(m2_merged_fouth_layer_1)
    m2_dense_fifth_layer_2 = Dense(30, activation='relu')(m2_merged_fouth_layer_2)

    m2_merge_fifth_layer = Concatenate()([m2_dense_fifth_layer_1, m2_dense_fifth_layer_2])

    m2_dense_final_layer = Dense(9, activation='relu')(m2_merge_fifth_layer)

    DMN = Model(inputs=[m2_input_layer, m2_input_layer_2, m2_input_layer_3, m2_input_layer_4, m2_input_layer_5,
                           m2_input_layer_6, m2_input_layer_7, m2_input_layer_8, m2_input_layer_9, m2_input_layer_10,
                           m2_input_layer_11, m2_input_layer_12, m2_input_layer_13, m2_input_layer_14,
                           m2_input_layer_15,
                           m2_input_layer_16, m2_input_layer_17, m2_input_layer_18, m2_input_layer_19,
                           m2_input_layer_20, m2_input_layer_21,
                           m2_input_layer_22, m2_input_layer_23, m2_input_layer_24, m2_input_layer_25,
                           m2_input_layer_26,
                           m2_input_layer_27, m2_input_layer_28, m2_input_layer_29, m2_input_layer_30,
                           m2_input_layer_31,
                           m2_input_layer_32],
                   outputs=m2_dense_final_layer, name="DMN")
    # DMN.save_weights("model4_initial_weights.h5")
    DMN.summary()

    DMN.compile(loss="mean_squared_error",
                optimizer='rmsprop',
                metrics=['accuracy']
                )

    # 模型的断点续传部分
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
    # Base learning rate after warmup.
    learning_rate_base = 0.0001

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
                x_valid_scaled_wide, x_valid_scaled_deep], y_valid),
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=[tensorboard, checkpoint, warm_up_lr])

    return history
    # DMN.evaluate(x_test,  y_test, verbose=2)