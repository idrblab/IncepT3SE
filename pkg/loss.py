import tensorflow as tf
from tensorflow.keras import backend as K

def focal_loss(gamma=[2,2], alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))   # pos
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))  # neg
        return -K.mean(1000 * alpha * K.pow(1. - pt_1, gamma[1]) * K.log(pt_1 + K.epsilon()))\
               -K.mean(1000 * (1 - alpha) * K.pow(pt_0, gamma[0]) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed
