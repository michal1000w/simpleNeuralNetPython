import time
import Matrix
import NeuralNetwork
from Import import Think_File
from colorama import Fore, Back, Style

class NeuNet:
    def __init__(self,experimental=False,sigm = True):
        self.neural_net = NeuralNetwork.NeuralNetwork(1,1,1,0,0,[],[],Matrix.Matrix(""))
        self.training_inputs = Matrix.Matrix("")
        self.training_outputs = Matrix.Matrix("")

        self.test_inputs = []
        self.test_outputs = []

        self.ID = Fore.CYAN + "[NeuNet] " + Fore.RESET
        self.sigm = sigm
        self.iteration = 0
        self.setup = False
        self.SEED = 1
        self.threads = 0
        self.experimental = experimental
        self.force = False
        self.print_synaptic = True
        self.arythmetic_mean = False

        #self.weights = Matrix.Matrix("")
        self.weights = []

        self.names = ""
        self.device = 0

        #multilayer
        self.hidden_Layout = Matrix.Matrix("")

        print(self.ID,"Created instance")

    def input(self,training_data:str):
        print(self.ID,"Adding training input data",flush=True)
        start = time.perf_counter()

        self.training_inputs.add(training_data)

        if (self.sigm):
            print(self.ID,"Using sigmoid function on training input data")
            self.training_inputs = self.training_inputs.sigmoid()
        
        print(self.ID, Fore.GREEN + "Done" + Fore.RESET + " in: ",(time.perf_counter() - start),"s",flush = True)

    def output(self,training_output:str):
        print(self.ID,"Adding training output data",flush=True)
        start = time.perf_counter()

        self.training_outputs.add(training_output)
        self.training_outputs = self.training_outputs.T()

        print(self.ID, Fore.GREEN + "Done" + Fore.RESET + " in: ",(time.perf_counter() - start),"s",flush = True)

    def print_synaptic_set(self,printS:bool):
        self.print_synaptic = printS
        self.neural_net.print_synaptic_set(printS)

    def labels(self,names:str):
        print(self.ID,"Adding labels")
        self.neural_net.add_names(names)
        self.names = names

    def load_synaptic_weights(self,weights:[]):
        print("Loading synaptic weights...")
        self.weights = weights
        self.neural_net.load_synaptic_weights(weights)

    def print_synaptic_weights(self):
        print("Synaptic weights:")
        self.neural_net.print_synaptic_weights()

    def set_name(self,name:str):
        self.neural_net.set_name(name)

    def set_device(self,device:str):
        dev = 0
        print("Training device set to: ",end="")
        if (device == "cpu"):
            dev = 0
            print("CPU")
        elif (device == "gpu"):
            dev = 1
            print("GPU")
        else:
            dev = 0
            print("CPU")
            print("Niepoprawny parametr:",device,"\nDo wyboru: \"cpu\"  |  \"gpu\"")
        print("\n")
        self.device = dev
        self.neural_net.set_device(dev)


    def iterations(self,iter:int):
        self.iteration = abs(iter)

    def seed(self,SEED):
        self.SEED = SEED

    def set_threads(self,threads):
        self.threads = threads
    
    def force_threads(self,force):
        self.force = force
    
    def force_arythmetic_mean(self,force):
        if (force):
            print(self.ID , Fore.RED + "Using arythmetic mean (may be less accurate)" + Fore.RESET)
        self.arythmetic_mean = force
        self.neural_net.force_arythmetic_mean(force)

    def force_float_reduction(self,force):
        if (force):
            print(self.ID , Fore.RED + "Reducing float accuracy (may be less accurate)" + Fore.RESET)
        self.neural_net.force_float_reduction(not(force))
    
    def go_experimental(self,experimental:bool):
        self.experimental = experimental

    def add_testing_data(self,test_inputs:[],test_outputs:[]):
        self.test_inputs = test_inputs

        if (self.sigm):
            for i in range(len(test_inputs)):
                self.test_inputs[i] = self.test_inputs[i].sigmoid()

        self.test_outputs = test_outputs

    def add_hidden_layout(self,layout:Matrix.Matrix):
        print(self.ID,"Adding hidden layout")
        self.hidden_Layout = layout

    def set_iter(self,iter:int):
        self.neural_net.set_iter(iter)

    def Setup(self):
        print(self.ID,"Starting Setup...")
        if (not(self.training_inputs.mat == [] or self.training_outputs.mat == [] or self.iteration == 0 or len(self.neural_net.nazwy) == 0 or self.test_inputs == [] or self.test_outputs == [])):
            print(self.ID,"Setting up NeuralNetwork")

            if(self.experimental):
                print(self.ID,Fore.RED + "!!!!Experimental mode enable (may cause some bugs)!!!!" + Fore.RESET)

            
            self.neural_net = NeuralNetwork.NeuralNetwork(self.training_inputs.kolumny,self.training_outputs.kolumny,self.SEED,self.threads,self.force,self.test_inputs,self.test_outputs,self.hidden_Layout,self.print_synaptic)
            self.neural_net.add_names(self.names)

            if (self.print_synaptic):
                print(self.ID,"Random generated synaptic weights:")
                self.neural_net.print_synaptic_weights()
            #self.neural_net.print_parameter_b()
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

            if (self.experimental):
                if (self.device == 0):
                    self.neural_net.train_server(self.training_inputs,self.training_outputs,self.iteration)
                else:
                    #self.neural_net.CUDA_train_Server(self.training_inputs,self.training_outputs,self.iteration)
                    self.neural_net.CUDA_train_Server_Fast(self.training_inputs,self.training_outputs,self.iteration)
            else:
                self.neural_net.train(self.training_inputs,self.training_outputs,self.iteration)

            durationTh = (time.perf_counter() - start_time)
            print(self.ID,Fore.GREEN + "Succeeded" + Fore.RESET + " in time: [ ",end="")
            if (durationTh < 60):
                print(durationTh,"] s \n")
            else:
                minutes = int(durationTh / 60)
                seconds = durationTh % 60
                print(str(minutes) + " min " + str(seconds),"s ] \n")

            if (self.print_synaptic):
                print(self.ID,"New synaptic weights after training: ")
                self.neural_net.print_synaptic_weights()
            print("\n")

            return 1
        else:
            print(self.ID,Fore.RED + "Training failed. (Try Setup first)" + Fore.RESET)
            return 0
        return 0

    def Think(self,data:str,realOutput = ""):
        print(self.ID,"Considering new situation:",data)
        nowa = Matrix.Matrix("")
        nowa.add(data)
        rOut = Matrix.Matrix("")
        rOut.add(realOutput)
        if (self.sigm):
            nowa = nowa.sigmoid()
        self.neural_net.think(nowa,rOut).printMatrix(0)
        #self.neural_net.print_classified()
        self.neural_net.print_classified_new()
        print("")

    def Think_from_File(self,test_inputs:[],test_outputs:[],filename = "untitled",labels_matrix = []):
        print(self.ID,"Testing all data...")

        inp = []
        for i in range(len(test_inputs)):
            inp.append(Matrix.Matrix(test_inputs[i].getString()))

        if (self.sigm):
            for i in range(len(test_inputs)):
                test_inputs[i] = test_inputs[i].sigmoid()
        loss = self.neural_net.test_loss(test_inputs,test_outputs)
        print("Loss: ",loss)
        print("\n\n")

        if (self.device == 0):
            wyniki = self.neural_net.test_training(test_inputs,test_outputs)
        else:
            wyniki = self.neural_net.test_training_cuda(test_inputs,test_outputs)

        tf = Think_File(filename)
        tf.save_think_output(inp,test_outputs,wyniki,self.names,labels_matrix)

        print("\n\n")