import tensorflow as tf

# This class is designed for recurrent neural network.
class RecurrentLoss():
    def RecurrentMeanSquared(output,target,batch_size):
        if batch_size != None :
            return tf.reduce_sum(0.5 * tf.pow(output-target,2)) / tf.constant([batch_size],dtype=tf.float64) 
        else :
            print('Warning, you should give a batch size. Since there''s no batch size, the loss would be large.')
            return tf.reduce_sum(0.5 * tf.pow(output-target,2))
        
