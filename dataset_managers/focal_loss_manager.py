# dataset_managers/focal_loss_manager.py

from dataset_managers.base_dataset_manager import BaseDatasetManager
import tensorflow as tf
from keras import backend as K


class DatasetManagerWithFocalLoss(BaseDatasetManager):
    def focal_loss(self, gamma=2., alpha=4.):
        def focal_loss_fixed(y_true, y_pred):
            epsilon = K.epsilon()
            y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            cross_entropy = -y_true * tf.math.log(y_pred)
            weight = alpha * y_true * K.pow((1 - y_pred), gamma)
            loss = weight * cross_entropy
            return K.mean(K.sum(loss, axis=-1))
        return focal_loss_fixed
