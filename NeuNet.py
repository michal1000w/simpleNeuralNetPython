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
        self.SEED = 1

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

    def seed(self,SEED):
        self.SEED = SEED

    def Setup(self):
        print(self.ID,"Starting Setup...")
        if (not(self.training_inputs.mat == [] or self.training_outputs.mat == [] or self.iteration == 0 or len(self.neural_net.nazwy) == 0)):
            print(self.ID,"Setting up NeuralNetwork")
            neur = NeuralNetwork.NeuralNetwork(self.training_inputs.kolumny,self.training_outputs.kolumny,self.SEED)
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

            #self.neural_net.train(self.training_inputs,self.training_outputs,self.iteration)
            self.neural_net.train_server(self.training_inputs,self.training_outputs,self.iteration)

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
        self.neural_net.think(nowa).printMatrix(0)
        self.neural_net.print_classified()
        print("")