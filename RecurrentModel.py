import tensorflow as tf
import RecurrentLoss as RL

class ReccurentModel():
    def __init__(self,optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),loss_fun=RL.RecurrentLoss.RecurrentMeanSquared,dtype=tf.float64):
        self.sess = tf.Session()
        self.optimizer = optimizer
        self.dtype = dtype
        self.input = tf.placeholder(dtype=self.dtype,shape=[None,None,None])
        self.target = tf.placeholder(dtype=self.dtype,shape=[None,None,None])
        self.layers = list()
        self.num_layers = 0
        self.loss_fun = loss_fun
        
    
        
    def Build(self,recurrentunit):
        self.layers.append(recurrentunit)
        self.num_layers += 1
        
    def _Initialize(self,output_dim,recurrentunit):
        recurrentunit.input = self.output
        recurrentunit.Initialize(output_dim)
        self.output = recurrentunit.output
        return int(self.output.shape[2])
    def _Initialize_Variables(self,input_dim):
        unit = self.layers[0]
        unit.input = self.input
        unit.Initialize(input_dim)
        self.output = unit.output
        input_dim = int(unit.output.shape[2])
        for unit in self.layers[1:] :
            input_dim = self._Initialize(input_dim,unit)
            
    def Fit(self,X_train,Y_train,num_steps=5000,clip=False,decay=False,**kwargs):
        self.batch_size = int(X_train.shape[0])
        self._Initialize_Variables(int(X_train.shape[2]))
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
        
        
            
        
    