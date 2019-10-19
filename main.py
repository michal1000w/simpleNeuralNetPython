import random
import time

import Matrix as Matrix

class NeuralNetwork:
    def __init__(self,neuron_inputs:int,neuron_count:int,seed:int):
        random.seed(seed)
        self.neuron_count = neuron_count
        self.neuron_inputs = neuron_inputs

        self.synaptic_weights = Matrix.Matrix("")



#main

m = Matrix.Matrix("[1,6,5][2,4,3]")
#m.print()
m.printMatrix()

#m.T().printMatrix()
#m.exp().printMatrix()
#m.exp(True).printMatrix()

#m.sigmoid().printMatrix()
#m.sigmoid_derivative().printMatrix()

'''c = Matrix("[2,3,6,7][4,5,8,9]")
(c-m).printMatrix()

(m + m).printMatrix()
(m - m).printMatrix()'''
(m*m).printMatrix()
(m*m.T()).printMatrix()