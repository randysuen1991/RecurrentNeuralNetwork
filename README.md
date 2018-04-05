# Reccurent Neural Network
This file consists of the basic elements needed to implement RNN.
These codes are designed for the first, third, and the last type of the following graph.
![](https://github.com/randysuen1991/Recurrent-Neural-Network/blob/master/figures/different_type.jpg)

The AddNumExample.py refers to the following website:<br>
https://www.data-blogger.com/2017/08/27/gru-implementation-tensorflow/.



## The Task: Adding Numbers

In the code example, a simple task is used for testing the GRU. Given two numbers a and b, their sum is computed: c = a + b. The numbers are first converted to reversed bitstrings. The reversal is also what most people would do by adding up two numbers. You start at the right from the number and if the sum is larger than  10, you carry (memorize) a certain number. The model is capable of learning what to carry. As an example, consider the number a = 3 and b = 1. In bitstrings (of length 3), we have a = [0, 1, 1] and b = [0, 0, 1]. In reversed bitstring representation, we have that a = [1, 1, 0] and b = [1, 0, 0]. The sum of these numbers is c = [0, 0, 1] in reversed bitstring representation. This is [1, 0, 0] in normal bitstring representation and this is equivalent to 4. These are all the steps which are also done by the code automatically.

The following three figures are the loss of three types of recurrent units: fully gated GRU, simplified GRU, LSTM and Vanilla RNN. We could find that all of them could learn how to add number.

![](https://github.com/randysuen1991/Recurrent-Neural-Network/blob/master/figures/full.png)
![](https://github.com/randysuen1991/Recurrent-Neural-Network/blob/master/figures/part.png)
![](https://github.com/randysuen1991/Recurrent-Neural-Network/blob/master/figures/lstm.png)
![](https://github.com/randysuen1991/Recurrent-Neural-Network/blob/master/figures/vanilla.png)
