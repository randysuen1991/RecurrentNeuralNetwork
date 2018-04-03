import tensorflow as tf
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import RecurrentUnit

def example1():
    
    
    def as_bytes(num, final_size):
    
        res = []
        for _ in range(final_size):
            res.append(num % 2)
            num //= 2
        return res

    def generate_example(num_bits):
    
        a = random.randint(0, 2**(num_bits - 1) - 1)
        b = random.randint(0, 2**(num_bits - 1) - 1)
        res = a + b
        return (as_bytes(a,  num_bits),
                as_bytes(b,  num_bits),
                as_bytes(res,num_bits))

    def generate_batch(num_bits, batch_size):
    
        x = np.empty((batch_size, num_bits, 2))
        y = np.empty((batch_size, num_bits, 1))

        for i in range(batch_size):
            a, b, r = generate_example(num_bits)
            x[i, :, 0] = a
            x[i, :, 1] = b
            y[i, :, 0] = r
            
        return x, y

    # Configuration
    batch_size = 100
    time_size = 5

    # Generate a test set and a train set containing 100 examples of numbers represented in 5 bits
    X_train, Y_train = generate_batch(time_size, batch_size)
    X_test, Y_test = generate_batch(time_size, batch_size)
    
    
    
    input_dims = 2
    hidden_size = 6
    
    sess = tf.Session()
    gru = RecurrentUnit.VanillaRecurrentUnit(input_dims,hidden_size)
    
    
    W = tf.Variable(dtype=tf.float64,initial_value=tf.truncated_normal(shape=(hidden_size,1),dtype=tf.float64,mean=0,stddev=0.1),name="w")
    B = tf.Variable(dtype=tf.float64,initial_value=tf.truncated_normal(shape=(1,),dtype=tf.float64,mean=0,stddev=0.1),name="b")
    output = tf.map_fn(fn = lambda h_t: tf.matmul(h_t, W) + B, elems = gru.h_t)

    expected_output = tf.placeholder(dtype=tf.float64,shape=(batch_size,time_size,1),name="expected_output")
    
    loss = tf.reduce_sum(0.5 * tf.pow(output - expected_output,2)) / float(batch_size)
    
    train_step = tf.train.AdamOptimizer().minimize(loss)
    
    init_variables = tf.global_variables_initializer()
    sess.run(init_variables)
    
    train_losses = []
    validation_losses = []
    
    for epoch in range(5000):
        
        _, train_loss = sess.run([train_step,loss],feed_dict={gru.input_layer:X_train,expected_output:Y_train})
        
        validation_loss = sess.run(loss,feed_dict={gru.input_layer:X_test,expected_output:Y_test})
        
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        
        # Display an update every 50 iterations
        if epoch % 50 == 0:
            plt.plot(train_losses, '-b', label='Train loss')
            plt.plot(validation_losses, '-r', label='Validation loss')
            plt.legend(loc=0)
            plt.title('Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.show()
            print('Iteration: %d, train loss: %.4f, test loss: %.4f' % (epoch, train_loss, validation_loss))
    # Define two numbers a and b and let the model compute a + b
    a = 1024
    b = 16

# The model is independent of the sequence length! Now we can test the model on even longer bitstrings
    bitstring_length = 20
    
# Create the feature vectors    
    X_custom_sample = np.vstack([as_bytes(a, bitstring_length), as_bytes(b, bitstring_length)]).T
    X_custom = np.zeros((1,) + X_custom_sample.shape)
    X_custom[0, :, :] = X_custom_sample

# Make a prediction by using the model
    y_predicted = sess.run(output, feed_dict={gru.input_layer: X_custom})
# Just use a linear class separator at 0.5
    y_bits = 1 * (y_predicted > 0.5)[0, :, 0]
# Join and reverse the bitstring
    y_bitstr = ''.join([str(int(bit)) for bit in y_bits.tolist()])[::-1]
# Convert the found bitstring to a number
    y = int(y_bitstr, 2)

# Print out the prediction
    print(y) # Yay! This should equal 1024 + 16 = 1040
    
    
if __name__=="__main__":
    example1()