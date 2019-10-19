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

    def print_names(self):
        print("Stored names: ", end="")
        for i in range(len(self.nazwy)):
            print("[",self.nazwy[i],"]",end="")
        print("")

    def add_names(self,data:str):
        #podział na pojedyńcze macierze (fragmenty)
        length = len(data)
        fragment = ""

        for i in range(length):
            if (data[i] == '['):
                fragment = ""
                while (i < length - 1):
                    i += 1
                    if (data[i] == ']'):
                        break
                    fragment += data[i]
                self.nazwy.append(fragment)

    #trenowanie sieci
    def train(self,training_inputs:Matrix.Matrix,training_outputs:Matrix.Matrix,iterations:int):
        output = Matrix.Matrix("")
        error = Matrix.Matrix("")
        adjustment = Matrix.Matrix("")

        modulo = 5 * (iterations / 100)

        print(" [ ",end="")

        for i in range(iterations):
            if (i%modulo == 0):
                print(str((i*100)/iterations)+"% ",end="",flush=True)
            
            output = self.think(training_inputs)
            error = training_outputs - output

            adjustment = training_inputs.T() * (output.sigmoid_derivative() ** error)

            self.synaptic_weights += adjustment
        
        print(" 100% ]")

    def think(self,inputs:Matrix.Matrix):
        self.wynik = (inputs * self.synaptic_weights).sigmoid()
        return self.wynik


#main

n = NeuralNetwork(3,2,1)
n.print_synaptic_weights()

n.add_names("[piesek][kotek]")
n.print_names()

inputs = Matrix.Matrix("[1,2,3][1,2,3][2,3,4]")
output = Matrix.Matrix("[1,1,0][0,0,1]").T()

n.train(inputs,output,1000)


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