import random
import time

import Matrix as Matrix

class NeuralNetwork:
    def __init__(self,neuron_inputs:int,neuron_count:int,seed:int):
        random.seed(seed)
        self.neuron_count = neuron_count
        self.neuron_inputs = neuron_inputs

        self.synaptic_weights = Matrix.Matrix("")
        self.wynik = Matrix.Matrix("")
        self.nazwy = []

        weights = ""
        liczba = ""
        for j in range(neuron_count):
            weights += '['
            for i in range(neuron_inputs):
                liczba = str((random.random() * 2) - 1)
                for z in range(len(liczba)):
                    weights += liczba[z]
                if (i == neuron_count):
                    weights += ' '
                else:
                    weights += ','
            weights += ']'

        self.synaptic_weights.add(weights)
        self.synaptic_weights = self.synaptic_weights.T()
    
    def print_synaptic_weights(self):
        self.synaptic_weights.printMatrix()

    def print_classified(self):
        klasy = self.neuron_count

        for i in range(klasy):
            if (self.wynik.getArray()[0][i] >= 0.5):
                print("[",self.nazwy[i],"]", end="")
        print("")



#main

n = NeuralNetwork(3,2,1)
n.print_synaptic_weights()

'''m = Matrix.Matrix("[1,6,5][2,4,3]")
#m.print()
m.printMatrix()

#m.T().printMatrix()
#m.exp().printMatrix()
#m.exp(True).printMatrix()

#m.sigmoid().printMatrix()
#m.sigmoid_derivative().printMatrix()

c = Matrix("[2,3,6,7][4,5,8,9]")
(c-m).printMatrix()

(m + m).printMatrix()
(m - m).printMatrix()
(m*m).printMatrix()
(m*m.T()).printMatrix()'''