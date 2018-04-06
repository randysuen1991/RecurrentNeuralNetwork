import tensorflow as tf

# Define a decorator that is used to check if the batch size is specified.
def BatchSizeCheck(fun):
    def decofun(**kwargs):
        if kwargs.get('batch_size',None) == None :
            print('Warning, you should give a batch size. Since there''s no batch size, the loss would be large.')
        return fun(kwargs.get('output'),kwargs.get('target'),kwargs.get('batch_size',1))
    return decofun 


# This class is designed for recurrent neural network.
class RecurrentLoss():
    @BatchSizeCheck
    def M2M_RecurrentMeanSquared(output,target,batch_size):
        return tf.reduce_sum(0.5 * tf.pow(output-target,2)) / tf.constant([batch_size],dtype=tf.float64) 

