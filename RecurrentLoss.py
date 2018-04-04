import tensorflow as tf

# This class is designed for recurrent neural network.
class RecurrentLoss():
    def RecurrentMeanSquared(h,Y):
        return tf.reduce_sum(0.5 * tf.pow(h-Y,2)) / tf.float64(h.shape[0])
