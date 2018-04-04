import tensorflow as tf
import RecurrentLoss as RL

class ReccurentModel():
    def __init__(self,optimizer=tf.train.AdamOptimizer,loss=RL.RecurrentLoss.RecurrentMeanSquared,dtype=tf.float64):
        self.sess = tf.Session()
        self.optimizer = optimizer
        self.dtype = dtype
        # The first None is batch size, the second one is the time step and the last is the input dimensions.
        self.target = tf.placeholder(dtype=self.dtype,shape=[None,None,None])
        self.layers = list()
        self.num_layers = 0
        #self.loss should be a function.
        self.loss = loss
    
    
    # arguement 'recurrentunit' should be the object in RecurrentUnit.py
    # Still needs to store the parameters in the model
    def Build(self,recurrentunit):
        if self.num_layers == 0 :
            if recurrentunit.input_dim == None :
                raise ValueError('The first layer should specify the input dimension')
            self.input = tf.placeholder(dtype=self.dtype,shape=[None,None,recurrentunit.input_dim])
            self.output = self.input
            recurrentunit.Initialize(recurrentunit.input_dim)
        else:
            recurrentunit.Initialize(int(self.output.shape[2]))
        
        self.layers.append(recurrentunit)
        recurrentunit.input = self.output
        self.output = recurrentunit.output
        
        self.num_layers += 1
        
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
        
        
            
        
    