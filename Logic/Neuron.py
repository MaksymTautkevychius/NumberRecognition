import random
import math
class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.01, 0.01)
        self.output = 0.0
        self.delta = 0.0

    def activate(self, x):
        return 1 / (1 + math.exp(-x))  # Sigmoid 

    def calculate_output(self, inputs):
        self.inputs = inputs
        net_input = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = self.activate(net_input)
        return self.output