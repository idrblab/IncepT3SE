import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import MaxPool1D, MaxPool2D, GlobalMaxPool2D, BatchNormalization
from tensorflow.keras.layers import Conv1D, Conv2D, Concatenate, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2, l1_l2


def Inception(inputs, filters, strides=1):  # inception block
    x1 = Conv1D(filters, kernel_size=5, padding='same', activation='relu', strides=strides, kernel_regularizer=l1_l2(0.003,0.003))(inputs)
    x2 = Conv1D(filters, kernel_size=3, padding='same', activation='relu', strides=strides, kernel_regularizer=l1_l2(0.003,0.003))(inputs)
    x3 = Conv1D(filters, kernel_size=1, padding='same', activation='relu', strides=strides, kernel_regularizer=l1_l2(0.003,0.003))(inputs)
    outputs = Concatenate()([x1, x2, x3])    
    return outputs

def T3Net(input_shape,  
              n_outputs,
              dense_layers = [128, 32],
              dense_avf = 'relu',
              last_avf = None):
    tf.keras.backend.clear_session()
    assert len(input_shape) == 3
    inputs = Input(input_shape)

    c_lst = []
    for i in range(3):
        mk = 3*i+3
        conv = Conv2D(1024, kernel_size=(mk,20), padding='valid', activation='relu', strides=1, kernel_regularizer=l1_l2(0.001,0.001))(inputs)
        conv = MaxPool2D(pool_size=5, strides=5, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = Dropout(0.5)(conv)
        incept = Inception(conv, filters=32)
        incept = MaxPool2D(pool_size=5, strides=5, padding='same')(incept)
        incept = Inception(incept, filters=64)
        c_lst.append(incept)
    x = Concatenate(axis=3)([c_lst[0],c_lst[1],c_lst[2]])
    x = GlobalMaxPool2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    for units in dense_layers:
        x = Dense(units, activation=dense_avf, kernel_regularizer=l1_l2(0.003,0.003))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
    outputs = Dense(n_outputs,activation=last_avf, kernel_regularizer=l1_l2(0.003,0.003))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# model = T3Net((200, 20, 1),
#                 n_outputs=2,
#                 dense_layers=[256, 32],
#                 dense_avf='relu',
#                 last_avf='softmax')      
# model.summary()
#
# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file="model.png", show_shapes=True)
