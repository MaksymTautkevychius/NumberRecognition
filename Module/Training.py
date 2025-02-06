import numpy as np 
import csv,math
import Reader,Neuron

def calculate_sigmoid(weights,inputs,bias):
    net=0
    for i in range(len(weights)):
        net=weights[i]*inputs[i]
    net-=bias
    sigmoid=1/(1+math.exp(net))
    print(sigmoid+'sigmoid value')
    return sigmoid


def Initialize_Neurons_For_Layer( layer: np.array):
    for i in range(len(layer)):
        layer[i] = Neuron(Neuron.Initialize_Weights(len(layer)), Neuron.Initialize_Biases(np.random.random(len(layer))))
        return layer
                
def Initialize_Neurons_For_Layer_Output(layer: np.array):                       
    for item in range(len(layer)):
        layer[item]=Neuron(Neuron.Initialize_Weights(len(layer)), Neuron.Initialize_Biases(np.random.random(len(layer))),int(item+1))
        return layer
    
def train(layers):
    return


def Create_layers(self,actual_numbers,train_data_list):
    Input_Layer=np.empty(len(train_data_list))
    Layer256=np.empty(256)
    Layer128=np.empty(128)
    Layer64=np.empty(64)
    Layer32=np.empty(32)
    OutputLayer=np.empty(10)
    Initialize_Neurons_For_Layer(Layer256)
    Initialize_Neurons_For_Layer(Layer128)
    Initialize_Neurons_For_Layer(Layer64)
    Initialize_Neurons_For_Layer(Layer32)
    Initialize_Neurons_For_Layer_Output(OutputLayer)
    layers=np.array([Input_Layer,Layer256,Layer128,Layer64,Layer32,OutputLayer])
    train(layers)
    
def main():
    train_data_list,actual_numbers=Reader.read_file('mnist_train.csv')
    test_data_list,actual_numbers_test=Reader.read_file('mnist_test.csv')
main()