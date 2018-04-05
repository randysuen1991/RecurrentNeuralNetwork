import tensorflow as tf
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import RecurrentModel as RM
import RecurrentLoss as RL
import RecurrentUnit as RU
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

    
    hidden_size = 6
    
    
    model = RM.ReccurentModel()
    model.Build(recurrentunit=RU.GatedRecurrentUnit(hidden_dim=hidden_size))
    model.Build(recurrentunit=RU.LongShortTermMemory(hidden_dim=hidden_size))
    model.Build(recurrentunit=RU.NeuronLayer(hidden_dim=1))
    model.Fit(X_train,Y_train,num_steps=5000,show_graph=True)
    

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
    y_predicted = model.sess.run(model.output, feed_dict={model.input: X_custom})
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