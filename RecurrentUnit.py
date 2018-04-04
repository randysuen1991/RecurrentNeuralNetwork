import tensorflow as tf


class RecurrentUnit():
    def __init__(self,input_dims,hidden_dims,dtype=tf.float64):
        self.dtype = dtype
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        # The first None is the batch size and the second None is the time_step
        self.input_layer = tf.placeholder(dtype=dtype,shape=(None,None,self.input_dims),name='input')
        self.parameters = dict()
    def _Forward_Pass():
        raise NotImplementedError


class NeuronLayer(RecurrentUnit):
    def __init__(self,input_dims,hidden_dims,dtype=tf.float64):
        super().__init__(input_dims,hidden_dims,dtype)
        self.parameters['w'] = tf.Variable(dtype=tf.float64,initial_value=tf.truncated_normal(shape=(self.hidden_dims,),dtype=tf.float64,mean=0,stddev=0.1))
        self.parameters['b'] = tf.Variable(dtype=tf.float64,initial_value=tf.truncated_normal(shape=(1,),dtype=tf.float64,mean=0,stddev=0.1))
        self.h = tf.map_fn(fn=lambda h_t: tf.matmul(h_t,self.parameters['w'])+self.parameters['b'],elems = self.input_layers)

class VanillaRecurrentUnit(RecurrentUnit):
    def __init__(self,input_dims=None,hidden_dims,dtype=tf.float64):
        super().__init__(input_dims,hidden_dims,dtype)
        self.parameters['wi'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.input_dims,self.hidden_dims),mean=0,stddev=0.1))
        self.parameters['wh'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,self.hidden_dims),mean=0,stddev=0.1))
        self.parameters['b'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,),mean=0,stddev=0.1))

        self.x_transposed = tf.transpose(self.input_layer,[1,0,2])
        self.h_0 = tf.matmul(self.x_transposed[0,:,:],tf.zeros(dtype=self.dtype, shape=(self.input_dims, self.hidden_dims)))
        self.h_t_transposed =  tf.scan(fn=self._Forward_Pass,elems=self.x_transposed,initializer=self.h_0)
        self.h_t = tf.transpose(self.h_t_transposed,[1,0,2])
    def _Forward_Pass(self,h_tm1,x_t):
        return tf.tanh(tf.matmul(x_t,self.parameters['wi']) + tf.matmul(h_tm1,self.parameters['wh']) + self.parameters['b'])

class GatedRecurrentUnit(RecurrentUnit):
    def __init__(self,input_dims,hidden_dims,dtype=tf.float64,full=True):
        super().__init__(input_dims,hidden_dims,dtype)
        # forget get parameters
        self.parameters['wif'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.input_dims,self.hidden_dims),mean=0,stddev=0.1))
        self.parameters['whf'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,self.hidden_dims),mean=0,stddev=0.1))
        
        # output gate parameters
        self.parameters['wio'] =tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.input_dims,self.hidden_dims),mean=0,stddev=0.1))
        self.parameters['who'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,self.hidden_dims),mean=0,stddev=0.1))
        
        self.parameters['bf'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,),mean=0,stddev=0.1))
        self.parameters['bo'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,),mean=0,stddev=0.1))
            
        
        self.x_transposed = tf.transpose(self.input_layer,[1,0,2])
        
        self.h_0 = tf.matmul(self.x_transposed[0,:,:],tf.zeros(dtype=self.dtype, shape=(self.input_dims, hidden_dims)))
        
        if full :
            # reset gate parameters
            self.parameters['wir'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.input_dims,self.hidden_dims),mean=0,stddev=0.1))
            self.parameters['whr'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,self.hidden_dims),mean=0,stddev=0.1))
            self.parameters['br'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,),mean=0,stddev=0.1))
            self.h_t_transposed = tf.scan(fn=self._Forward_Pass_Full, elems = self.x_transposed, initializer=self.h_0)
        else :
            self.h_t_transposed = tf.scan(fn=self._Forward_Pass, elems = self.x_transposed, initializer=self.h_0)
        
        self.h_t = tf.transpose(self.h_t_transposed,[1,0,2])
            
    def _Forward_Pass_Full(self, h_tm1, x_t):
        f_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wif']) + tf.matmul(h_tm1,self.parameters['whf']) + self.parameters['bf'])
        r_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wir']) + tf.matmul(h_tm1,self.parameters['whr']) + self.parameters['br'])
        h_proposal = tf.tanh(tf.matmul(x_t, self.parameters['wio']) + tf.matmul(tf.multiply(r_t, h_tm1), self.parameters['who']) + self.parameters['bo'])
        h_t = tf.multiply(1-f_t,h_tm1) + tf.multiply(f_t,h_proposal)
        return h_t
    
    def _Forward_Pass(self, h_tm1, x_t):
        f_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wif']) + tf.matmul(h_tm1,self.parameters['whf']) + self.parameters['bf'])
        h_proposal = tf.tanh(tf.matmul(x_t, self.parameters['wio']) + tf.matmul(tf.multiply(f_t, h_tm1), self.parameters['who']) + self.parameters['bo'])
        h_t = tf.multiply(f_t,h_tm1) + tf.multiply(1-f_t,h_proposal)
        return h_t
    
class LongShortTermMemory(RecurrentUnit):
    def __init__(self,input_dims,hidden_dims,dtype=tf.float64):
        super().__init__(input_dims,hidden_dims,dtype)
        # 'w' means weights, 'i' in the middel place means input data and 'f' means forget gate.
        self.parameters['wif'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.input_dims,self.hidden_dims),mean=0,stddev=0.1),name='Wif')
        # 'h' means the info from last time.
        self.parameters['whf'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,self.hidden_dims),mean=0,stddev=0.1),name='Wif')
        # 'i' in the last place means input gate.
        self.parameters['wii'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.input_dims,self.hidden_dims),mean=0,stddev=0.1),name='Wif')
        self.parameters['whi'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,self.hidden_dims),mean=0,stddev=0.1),name='Wif')
        # 'o' in the last place means output gate.
        self.parameters['wio'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.input_dims,self.hidden_dims),mean=0,stddev=0.1),name='Wif')
        self.parameters['who'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,self.hidden_dims),mean=0,stddev=0.1),name='Wif')
        # 'c' means selection gate.
        self.parameters['wic'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.input_dims,self.hidden_dims),mean=0,stddev=0.1),name='Wif')
        self.parameters['whc'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,self.hidden_dims),mean=0,stddev=0.1),name='Wif')
            
        # 'b' means bias.
        self.parameters['bf'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,),mean=0,stddev=0.1),name='Wif')
        self.parameters['bi'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,),mean=0,stddev=0.1),name='Wif')
        self.parameters['bo'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,),mean=0,stddev=0.1),name='Wif')
        self.parameters['bc'] = tf.Variable(initial_value=tf.truncated_normal(dtype=dtype,shape=(self.hidden_dims,),mean=0,stddev=0.1),name='Wif')
        
        
        self.x_transposed = tf.transpose(self.input_layer,perm=[1,0,2])
        self.c_0_transposed = tf.matmul(self.x_transposed[0,:,:],tf.zeros(shape=(self.input_dims,self.hidden_dims),dtype=self.dtype))
        self.h_0 = tf.matmul(self.x_transposed[0,:,:],tf.zeros(shape=(self.input_dims,self.hidden_dims),dtype=self.dtype))
        self.h_t_transposed, self.c_t_transposed = tf.scan(fn=self.Forward_Pass,elems=self.x_transposed,initializer=(self.h_0,self.c_0_transposed))
        self.h_t = tf.transpose(self.h_t_transposed,perm=[1,0,2])
        
    def _Forward_Pass(self, info_tm1, x_t):
        h_tm1 = info_tm1[0]
        c_tm1 = info_tm1[1]
        
        f_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wif']) + tf.matmul(h_tm1,self.parameters['whf']) + self.parameters['bf'])
        i_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wii']) + tf.matmul(h_tm1,self.parameters['whi']) + self.parameters['bi'])
        o_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wio']) + tf.matmul(h_tm1,self.parameters['who']) + self.parameters['bo'])
        c_t = f_t * c_tm1 + i_t * tf.tanh(tf.matmul(x_t,self.parameters['wic']) + tf.matmul(h_tm1,self.parameters['whc'] + self.parameters['bc']))
        h_t = o_t * tf.tanh(c_t)
        
        return h_t, c_t

