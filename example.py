import tensorflow as tf
import RecurrentLoss as RL
import RecurrentModel as RM
import RecurrentUnit as RU


def example1():
    
    model = RM.ReccurentModel(batch_size=10)
    model.Build(recurrentunit=RU.GatedRecurrentUnit(input_dim=5,hidden_dim=7))
    model.Build(recurrentunit=RU.GatedRecurrentUnit(hidden_dim=9))
    model.Build(recurrentunit=RU.NeuronLayer(hidden_dim=1))
    
    
if __name__ == '__main__' :
    example1()