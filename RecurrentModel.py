import tensorflow as tf
import RecurrentLoss as RL

class ReccurentModel():
    def __init__(self,optimizer=tf.train.AdamOptimizer,loss=RL.RecurrentLoss.RecurrentMeanSquared,dtype=tf.float64):
        self.sess = tf.Session()
        self.optimizer = optimizer
        self.dtype = dtype
        # The first None is batch size and the second one is the time step
        self.input_layer = tf.placeholder(dtype=self.dtype,shape=[None,None,None])
        # Presume the output to be the input
        self.h = self.input_layer
        self.target = tf.placeholder(dtype=self.dtype,shape=[None,None,None])
        self.parameters = list()
        self.num_layers = 0
        #self.loss should be a function.
        self.loss = loss
    
    
    # arguement 'recurrentunit' should be the object in RecurrentUnit.py
    # Still needs to store the parameters in the model
    def Build(self,recurrentunit):
        self.num_layers += 1
        self.parameters.append(recurrentunit.parameters)
        recurrentunit.input_layer = self.h
        self.h = recurrentunit.h_t
        
    def Fit(self,X_train,Y_train,num_steps=100,clip=False,decay=False,**kwargs):
        loss = self.loss(X_train,Y_train)
        grads_and_vars = self.optimzer.compute_gradients(loss)
        if clip :
            grads, vars = zip(*grads_and_vars)
            grads, _ = tf.clip_by_norm(t=grads,clip_norm=kwargs.get('clip_norm',1.25))
            grads_and_vars = zip(grads,vars)
        train = self.optimizer.apply_gradients(grads_and_vars)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for i in range(num_steps):
            _, train_loss = self.sess.run(fetches=[train,loss],feed_dict={self.input_layer:X_train,self.target:Y_train})
    def Predict(self,X_test,Y_test=None):
        results = self.sess.run(fetch=[self.h,self.loss(X_test,Y_test)],fetch_dict={self.input_layer:X_test})
        return results
        
        
            
        
    