import tensorflow as tf
import RecurrentLoss as RL

class ReccurentModel():
    def __init__(self,batch_size=None,optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),loss_fun=RL.RecurrentLoss.RecurrentMeanSquared,dtype=tf.float64):
        self.sess = tf.Session()
        self.optimizer = optimizer
        self.dtype = dtype
        self.input = tf.placeholder(dtype=self.dtype,shape=[None,None,None])
#        self.output = self.input
        # The first None is batch size, the second one is the time step and the last is the input dimensions.
        self.target = tf.placeholder(dtype=self.dtype,shape=[None,None,None])
        self.layers = list()
        self.batch_size = batch_size
        self.num_layers = 0
        #self.loss_fun should be a function.
        self.loss_fun = loss_fun
    
    
    # arguement 'recurrentunit' should be the object in RecurrentUnit.py
    # Still needs to store the parameters in the model
    def Build(self,recurrentunit):
        if self.num_layers == 0 :
            if recurrentunit.input_dim == None :
                raise ValueError('The first layer should specify the input dimension')
            self.input = tf.placeholder(dtype=self.dtype,shape=[100,5,recurrentunit.input_dim])
            recurrentunit.input = self.input
            recurrentunit.Initialize(recurrentunit.input_dim)
        else:
            recurrentunit.input = self.output
            recurrentunit.Initialize(int(self.output.shape[2]))
        
        self.layers.append(recurrentunit)
        
        self.output = recurrentunit.output
        
        self.num_layers += 1
        
#    def Build(self,recurrentunit):
#        self.layers.append(recurrentunit)
#    def _Initialize(self,output_dim,recurrentunit):
#        recurrentunit.Initialize(output_dim)
#        recurrentunit.input = self.output
#        self.output = recurrentunit.output
#        return int(self.output.shape[2])
    
    def Fit(self,X_train,Y_train,num_steps=100,clip=False,decay=False,**kwargs):
#        layers2 = tf.tensor_shape(self.layers)
#        print(layers2)
#        _ = tf.scan(self._Initialize,self.layers2,initializer=X_train.shape[2])
        loss = self.loss_fun(self.output,self.target,self.batch_size)
        self.sess.run(tf.global_variables_initializer())
        grads_and_vars = self.optimizer.compute_gradients(loss)
        if clip :
            grads, vars = zip(*grads_and_vars)
            grads, _ = tf.clip_by_norm(t=grads,clip_norm=kwargs.get('clip_norm',1.25))
            grads_and_vars = zip(grads,vars)
        train = self.optimizer.apply_gradients(grads_and_vars)
        for i in range(num_steps):
            _, train_loss = self.sess.run(fetches=[train,loss],feed_dict={self.input:X_train,self.target:Y_train})
            print(train_loss)
        return train_loss
    def Predict(self,X_test,Y_test=None):
        results = self.sess.run(fetch=[self.output,self.loss_fun(X_test,Y_test)],fetch_dict={self.input_layer:X_test})
        return results
        
        
            
        
    