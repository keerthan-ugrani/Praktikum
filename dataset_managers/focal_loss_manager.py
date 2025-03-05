import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy_loss = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy_loss, axis=-1))
    return focal_loss_fixed
