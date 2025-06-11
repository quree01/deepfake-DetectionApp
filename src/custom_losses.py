# src/custom_losses.py
import tensorflow as tf
# Remove: from tensorflow import keras # This was causing the issue

# Use tf.keras.utils.register_keras_serializable for TensorFlow 2.x
@tf.keras.utils.register_keras_serializable(package="CustomLosses")
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss implementation as found in video_cdf/FocalLoss.py.
    Ensure gamma and alpha match the values used during model training.
    """
    epsilon = tf.keras.backend.epsilon() # Use tf.keras.backend instead of keras.backend
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # Calculate cross-entropy
    bce = y_true * tf.math.log(y_pred + epsilon)
    bce += (1 - y_true) * tf.math.log(1 - y_pred + epsilon)
    bce = -bce

    # Calculate pt (probability of correct class)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)

    # Calculate alpha factor
    alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)

    # Calculate focal loss
    focal_loss = alpha_factor * tf.pow(1. - pt, gamma) * bce

    return tf.reduce_mean(focal_loss)

# Important: If your model had other custom layers or metrics,
# you would define them here with their respective @tf.keras.utils.register_keras_serializable() decorators.