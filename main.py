import time

import Matrix
import NeuralNetwork

class NeuNet:
    def __init__(self,sigm = True):
        self.neural_net = NeuralNetwork.NeuralNetwork(1,1,1)
        self.training_inputs = Matrix.Matrix("")
        self.training_outputs = Matrix.Matrix("")

        self.ID = "[NeuNet] "
        self.sigm = sigm
        self.iteration = 0
        self.setup = False

        self.names = ""

        print(self.ID,"Created instance")

    def input(self,training_data:str):
        print(self.ID,"Adding training input data")
        self.training_inputs.add(training_data)

        if (self.sigm):
            print(self.ID,"Using sigmoid function on training input data")
            self.training_inputs = self.training_inputs.sigmoid()

    def output(self,training_output:str):
        print(self.ID,"Adding training output data")
        self.training_outputs.add(training_output)
        self.training_outputs = self.training_outputs.T()

    def labels(self,names:str):
        print(self.ID,"Adding labels")
        self.neural_net.add_names(names)
        self.names = names

    def iterations(self,iter:int):
        self.iteration = abs(iter)

    def Setup(self):
        print(self.ID,"Starting Setup...")
        if (not(self.training_inputs.mat == [] or self.training_outputs.mat == [] or self.iteration == 0 or len(self.neural_net.nazwy) == 0)):
            print(self.ID,"Setting up NeuralNetwork")
            neur = NeuralNetwork.NeuralNetwork(self.training_inputs.kolumny,self.training_outputs.kolumny,1)
            neur.add_names(self.names)
            self.neural_net = neur

            print(self.ID,"Random starting synaptic weights:")
            self.neural_net.print_synaptic_weights()
            print("")
            self.setup = True
            return 1

        else:
            print(self.ID,"Setup failed")
            return 0

    def Train(self):
        print(self.ID,"Starting Training...")
        if (self.setup):
            durationTh = 0.0
            print(self.ID,"Iterations:",self.iteration)
            start_time = time.perf_counter()

            self.neural_net.train(self.training_inputs,self.training_outputs,self.iteration)

            durationTh = (time.perf_counter() - start_time)
            print(self.ID,"Succeeded in time: [",durationTh,"] s \n")

            print(self.ID,"New synaptic weights after training: ")
            self.neural_net.print_synaptic_weights()
            print("\n")

            return 1
        else:
            print(self.ID,"Training failed. (Try Setup first)")
            return 0
        return 0

    def Think(self,data:str):
        print(self.ID,"Considering new situation:",data)
        nowa = Matrix.Matrix("")
        nowa.add(data)
        if (self.sigm):
            nowa = nowa.sigmoid()
        self.neural_net.think(nowa).printMatrix()
        self.neural_net.print_classified()
        print("")


#main

net = NeuNet()

net.input("[1,2,3][2,3,4]")
net.output("[1,0][0,1]")
net.iterations(1000)
net.labels("[ziemniak][kot]")
net.Setup()
net.Train()

net.Think("[1,3,5]")

'''n = NeuralNetwork.NeuralNetwork(3,2,1)
n.print_synaptic_weights()

n.add_names("[piesek][kotek]")
n.print_names()

inputs = Matrix.Matrix("[1,2,3][1,2,3][2,3,4]")
output = Matrix.Matrix("[1,1,0][0,0,1]").T()

n.train(inputs,output,1000)
m = Matrix.Matrix("[5,3,3]")
n.think(m)
n.print_classified()'''


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