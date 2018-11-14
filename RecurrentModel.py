import tensorflow as tf
import RecurrentLoss as RL
import matplotlib.pyplot as plt


class ReccurentModel():
    def __init__(self, dtype=tf.float64, **kwargs):
        self.graph = kwargs.get('graph', tf.Graph())
        self.sess = tf.Session(self.graph)
        self.loss_fun = None
        self.optimizer = None
        self.batch_size = None
        self.dtype = dtype
        # [None,None,None] = [batch_size,time_step,input_dim]
        self.input = tf.placeholder(dtype=self.dtype, shape=[None, None, None])
        self.target = tf.placeholder(dtype=self.dtype, shape=[None, None, None])
        self.layers = list()
        self.num_layers = 0
        
    def build(self, recurrentunit):
        self.layers.append(recurrentunit)
        self.num_layers += 1
        
    def _initialize(self, output_dim, recurrentunit):
        recurrentunit.input = self.output
        recurrentunit.initialize(output_dim)
        self.output = recurrentunit.output
        return int(self.output.shape[2])
    
    def _initialize_variables(self, input_dim):
        unit = self.layers[0]
        unit.input = self.input
        unit.initialize(input_dim)
        self.output = unit.output
        input_dim = int(unit.output.shape[2])
        for unit in self.layers[1:]:
            input_dim = self._initialize(input_dim, unit)
            
    def fit(self, x_train, y_train, num_steps=5000, optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1),
            loss_fun=RL.RecurrentLoss.RecurrentMeanSquared, clip=False, decay=False, show_graph=False, **kwargs):
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        # Here, we specify the batch size to be the number of training data
        self.batch_size = int(x_train.shape[0])
        self._initialize_variables(int(x_train.shape[2]))
        loss = self.loss_fun(output=self.output, target=self.target, batch_size=self.batch_size)

        self.sess.run(tf.global_variables_initializer())
        grads_and_vars = self.optimizer.compute_gradients(loss)
        if clip :
            grads, _vars = zip(*grads_and_vars)
            grads, _ = tf.clip_by_norm(t=grads, clip_norm=kwargs.get('clip_norm', 1.25))
            grads_and_vars = zip(grads, _vars)
        train = self.optimizer.apply_gradients(grads_and_vars)
        train_losses = list()
        for i in range(num_steps):
            _, train_loss = self.sess.run(fetches=[train, loss], feed_dict={self.input: x_train, self.target: y_train})
            train_losses.append(train_loss)

            if show_graph:
                # Display an update every 50 iterations
                if i % 50 == 0:
                    plt.plot(train_losses, '-b', label='Train loss')
                    plt.legend(loc=0)
                    plt.title('Loss')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')
                    plt.show()
                    print('Iteration: %d, train loss: %.4f' % (i, train_loss))
        return train_losses
    
    def predict(self, x_test, y_test=None):
        results = self.sess.run(fetches=[self.output, self.loss_fun(x_test, y_test)], feed_dict={self.input: x_test})
        return results
    