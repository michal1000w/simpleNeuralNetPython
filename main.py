import time

import Matrix
import NeuralNetwork

#main

n = NeuralNetwork.NeuralNetwork(3,2,1)
n.print_synaptic_weights()

n.add_names("[piesek][kotek]")
n.print_names()

inputs = Matrix.Matrix("[1,2,3][1,2,3][2,3,4]")
output = Matrix.Matrix("[1,1,0][0,0,1]").T()

n.train(inputs,output,1000)
m = Matrix.Matrix("[5,3,3]")
n.think(m)
n.print_classified()


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