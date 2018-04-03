import tensorflow as tf
import RecurrentUnit as RU


class Optimizer():
    pass





class ReccurentModel():
    def __init__(self,input_dims,optimizer,dtype=tf.float64):
        self.sess = tf.Session()
        self.optimizer = optimizer
        self.dtype = dtype
        self.input_dims = input_dims
        self.input_layer = tf.placeholder(dtype=self.dtype,shape=[None,None,input_dims])
        # Presume the output to be the input
        self.h = self.input
        self.target = tf.placeholder(dtype=self.dtype,shape=[None,None,input_dims])
        self.parameters = dict()
    
    def CollectParameters(self,func):
        def func_wrapped():
            pass
        return func_wrapped
    
    # arguement 'recurrentunit' should be the object in RecurrentUnit.py
    @CollectParameters
    def Build(self,recurrentunit):
        recurrentunit.input_layer = self.h
        self.h = recurrentunit.h
        
    