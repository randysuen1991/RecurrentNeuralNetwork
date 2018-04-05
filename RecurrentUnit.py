import tensorflow as tf


class RecurrentUnit():
    def __init__(self,hidden_dim,input_dim,dtype=tf.float64):
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.parameters = dict()
    def _Forward_Pass():
        raise NotImplementedError


class NeuronLayer(RecurrentUnit):
    def __init__(self,hidden_dim,input_dim=None,dtype=tf.float64):
        super().__init__(hidden_dim,input_dim)
    def Initialize(self,input_dim):
        self.input_dim = input_dim
        self.parameters['w'] = tf.Variable(initial_value=tf.truncated_normal(shape=(self.input_dim,1),dtype=tf.float64,mean=0,stddev=0.1))
        self.parameters['b'] = tf.Variable(initial_value=tf.truncated_normal(shape=(1,),dtype=tf.float64,mean=0,stddev=0.1))
        self.output = tf.map_fn(fn=lambda output: tf.matmul(output,self.parameters['w'])+self.parameters['b'],elems = self.input)

class VanillaRecurrentUnit(RecurrentUnit):
    def __init__(self,hidden_dim,input_dim=None,dtype=tf.float64):
        super().__init__(hidden_dim,input_dim)
    def Initialize(self,input_dim):
        self.input_dim = input_dim
        self.parameters['wi'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.input_dim,self.hidden_dim),mean=0,stddev=0.1))
        self.parameters['wh'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,self.hidden_dim),mean=0,stddev=0.1))
        self.parameters['b'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,),mean=0,stddev=0.1))

        self.input_transposed = tf.transpose(self.input,[1,0,2])
        self.h_0 = tf.matmul(self.input_transposed[0,:,:],tf.zeros(dtype=self.dtype, shape=(self.input_dim, self.hidden_dim)))
        self.output_transposed =  tf.scan(fn=self._Forward_Pass,elems=self.input_transposed,initializer=self.h_0)
        self.output = tf.transpose(self.output_transposed,[1,0,2])
    def _Forward_Pass(self,output_m1,x_t):
        return tf.tanh(tf.matmul(x_t,self.parameters['wi']) + tf.matmul(output_m1,self.parameters['wh']) + self.parameters['b'])

class GatedRecurrentUnit(RecurrentUnit):
    def __init__(self,hidden_dim,input_dim=None,dtype=tf.float64,full=True):
        super().__init__(hidden_dim,input_dim)
        self.full = full
    def Initialize(self,input_dim):
        self.input_dim = input_dim
        
        # forget get parameters
        self.parameters['wif'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.input_dim,self.hidden_dim),mean=0,stddev=0.1))
        self.parameters['whf'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,self.hidden_dim),mean=0,stddev=0.1))
        
        # output gate parameters
        self.parameters['wio'] =tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.input_dim,self.hidden_dim),mean=0,stddev=0.1))
        self.parameters['who'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,self.hidden_dim),mean=0,stddev=0.1))
        
        self.parameters['bf'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,),mean=0,stddev=0.1))
        self.parameters['bo'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,),mean=0,stddev=0.1))
            
        
        self.input_transposed = tf.transpose(self.input,[1,0,2])
        
        self.h_0 = tf.matmul(self.input_transposed[0,:,:],tf.zeros(dtype=self.dtype, shape=(self.input_dim, self.hidden_dim)))
        
        if self.full :
            # reset gate parameters
            self.parameters['wir'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.input_dim,self.hidden_dim),mean=0,stddev=0.1))
            self.parameters['whr'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,self.hidden_dim),mean=0,stddev=0.1))
            self.parameters['br'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,),mean=0,stddev=0.1))
            self.output_transposed = tf.scan(fn=self._Forward_Pass_Full, elems = self.input_transposed, initializer=self.h_0)
        else :
            self.output_transposed = tf.scan(fn=self._Forward_Pass, elems = self.input_transposed, initializer=self.h_0)
        
        self.output = tf.transpose(self.output_transposed,[1,0,2])
            
    def _Forward_Pass_Full(self, output_m1, x_t):
        f_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wif']) + tf.matmul(output_m1,self.parameters['whf']) + self.parameters['bf'])
        r_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wir']) + tf.matmul(output_m1,self.parameters['whr']) + self.parameters['br'])
        h_proposal = tf.tanh(tf.matmul(x_t, self.parameters['wio']) + tf.matmul(tf.multiply(r_t, output_m1), self.parameters['who']) + self.parameters['bo'])
        output = tf.multiply(1-f_t,output_m1) + tf.multiply(f_t,h_proposal)
        return output
    
    def _Forward_Pass(self, output_m1, x_t):
        f_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wif']) + tf.matmul(output_m1,self.parameters['whf']) + self.parameters['bf'])
        h_proposal = tf.tanh(tf.matmul(x_t, self.parameters['wio']) + tf.matmul(tf.multiply(f_t, output_m1), self.parameters['who']) + self.parameters['bo'])
        output = tf.multiply(f_t,output_m1) + tf.multiply(1-f_t,h_proposal)
        return output
    
class LongShortTermMemory(RecurrentUnit):
    def __init__(self,hidden_dim,input_dim=None,dtype=tf.float64):
        super().__init__(hidden_dim,input_dim)
    def Initialize(self,input_dim):
        self.input_dim = input_dim
        # 'w' means weights, 'i' in the middel place means input data and 'f' means forget gate.
        self.parameters['wif'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.input_dim,self.hidden_dim),mean=0,stddev=0.1),name='Wif')
        # 'h' means the info from last time.
        self.parameters['whf'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,self.hidden_dim),mean=0,stddev=0.1),name='Wif')
        # 'i' in the last place means input gate.
        self.parameters['wii'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.input_dim,self.hidden_dim),mean=0,stddev=0.1),name='Wif')
        self.parameters['whi'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,self.hidden_dim),mean=0,stddev=0.1),name='Wif')
        # 'o' in the last place means output gate.
        self.parameters['wio'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.input_dim,self.hidden_dim),mean=0,stddev=0.1),name='Wif')
        self.parameters['who'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,self.hidden_dim),mean=0,stddev=0.1),name='Wif')
        # 'c' means selection gate.
        self.parameters['wic'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.input_dim,self.hidden_dim),mean=0,stddev=0.1),name='Wif')
        self.parameters['whc'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,self.hidden_dim),mean=0,stddev=0.1),name='Wif')
            
        # 'b' means bias.
        self.parameters['bf'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,),mean=0,stddev=0.1),name='Wif')
        self.parameters['bi'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,),mean=0,stddev=0.1),name='Wif')
        self.parameters['bo'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,),mean=0,stddev=0.1),name='Wif')
        self.parameters['bc'] = tf.Variable(initial_value=tf.truncated_normal(dtype=self.dtype,shape=(self.hidden_dim,),mean=0,stddev=0.1),name='Wif')
        
        
        self.input_transposed = tf.transpose(self.input,perm=[1,0,2])
        self.c_0_transposed = tf.matmul(self.input_transposed[0,:,:],tf.zeros(shape=(self.input_dim,self.hidden_dim),dtype=self.dtype))
        self.h_0 = tf.matmul(self.input_transposed[0,:,:],tf.zeros(shape=(self.input_dim,self.hidden_dim),dtype=self.dtype))
        self.output_transposed, self.c_t_transposed = tf.scan(fn=self._Forward_Pass,elems=self.input_transposed,initializer=(self.h_0,self.c_0_transposed))
        self.output = tf.transpose(self.output_transposed,perm=[1,0,2])
        
    def _Forward_Pass(self, info_tm1, x_t):
        output_m1 = info_tm1[0]
        c_tm1 = info_tm1[1]
        
        f_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wif']) + tf.matmul(output_m1,self.parameters['whf']) + self.parameters['bf'])
        i_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wii']) + tf.matmul(output_m1,self.parameters['whi']) + self.parameters['bi'])
        o_t = tf.sigmoid(tf.matmul(x_t,self.parameters['wio']) + tf.matmul(output_m1,self.parameters['who']) + self.parameters['bo'])
        c_t = f_t * c_tm1 + i_t * tf.tanh(tf.matmul(x_t,self.parameters['wic']) + tf.matmul(output_m1,self.parameters['whc'] + self.parameters['bc']))
        output = o_t * tf.tanh(c_t)
        
        return output, c_t

