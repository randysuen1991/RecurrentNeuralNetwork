import tensorflow as tf
import RecurrentLoss as RL
import RecurrentModel as RM
import RecurrentUnit as RU


def example1():
    
    model = RM.ReccurentModel()
    model.Build(recurrentunit=RU.GatedRecurrentUnit(input_dim=5,hidden_dim=7))
    model.Build(recurrentunit=RU.GatedRecurrentUnit(hidden_dim=9))
    
    
if __name__ == '__main__' :
    example1()