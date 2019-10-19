import random
import Matrix

from multiprocessing import Process, Manager, Lock, Array
from multiprocessing.managers import BaseManager
import os
import time

class NeuralNetwork:
    def __init__(self,neuron_inputs:int,neuron_count:int,seed:int):
        random.seed(seed)
        self.SEED = seed
        self.neuron_count = neuron_count
        self.neuron_inputs = neuron_inputs

        self.synaptic_weights = Matrix.Matrix("")
        self.wynik = Matrix.Matrix("")
        self.nazwy = []

        self.synaptic_batches = []

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
            
            #algorytm start
            output = self.think(training_inputs)
            error = training_outputs - output

            adjustment = training_inputs.T() * (output.sigmoid_derivative() ** error)

            self.synaptic_weights += adjustment
        
        print(" 100% ]")

    def think(self,inputs:Matrix.Matrix):
        self.wynik = (inputs * self.synaptic_weights).sigmoid()
        return self.wynik

    #multithreading
    def synaptic_generator(self,ID,synaptic:Matrix.Matrix,neuron_count,neuron_inputs):
        random.seed(self.SEED + ID)

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

        synaptic.add(weights)
        synaptic = synaptic.T()

        random.seed(self.SEED)
        return synaptic


    def mtrain(self,training_inputs:Matrix.Matrix,training_outputs:Matrix.Matrix,iterations:int,lock:Lock,ID:int):
        output = Matrix.Matrix("")
        error = Matrix.Matrix("")
        adjustment = Matrix.Matrix("")

        synaptic_weights = Matrix.Matrix("")
        synaptic_weights = self.synaptic_generator(ID,synaptic_weights,self.neuron_count,self.neuron_inputs)

        #time.sleep(0.001*ID) #async

        for i in range(iterations):
            #algorytm start
            output = (training_inputs * synaptic_weights).sigmoid()
            error = training_outputs - output

            adjustment = training_inputs.T() * (output.sigmoid_derivative() ** error)

            synaptic_weights += adjustment

        lock.acquire()
        try:
            self.synaptic_batches[ID] = synaptic_weights
        finally:
            lock.release()

    def train_server(self,training_inputs:Matrix.Matrix,training_outputs:Matrix.Matrix,iterations:int):
        processes = []
        lock = Lock()

        #dzielenie na batche
        batches_in = []
        batches_out = []
        data_count = len(training_inputs.getArray())
        cpu_count = os.cpu_count()
        #cpu_count = 1

        if (data_count >= cpu_count):
            batch_size = int(data_count / cpu_count)
            left_size = data_count - cpu_count*batch_size
            cores_used = cpu_count
        else:
            batch_size = 1
            left_size = 0
            cores_used = data_count

        print("\n[Creating Batches]")
        print("CPU Threads: ",os.cpu_count())
        print("Data count: ",data_count)
        print("Batch size: ",batch_size)
        print("Left size: ",left_size)
        print("Threads used: ",cores_used)
        print("")

        #create synaptic batches
        #pamięć współdzielona
        manager = Manager()
        self.synaptic_batches = manager.list()
        
        for i in range(cores_used):
            self.synaptic_batches.append(self.synaptic_weights)


        #create batches [input data]
        data_length = len(training_inputs.getArray()[0])
        for i in range(cores_used): #batch_size * 
            mat = []
            for j in range(batch_size): #wiersze
                mat.append(training_inputs.getArray()[i * batch_size + j])
            new_batch = Matrix.Matrix("",batch_size,data_length,mat)
            batches_in.append(new_batch)

        #create batches [output data]
        data_length = len(training_outputs.getArray()[0])
        for i in range(cores_used):
            mat = []
            for j in range(batch_size): #wiersze
                mat.append(training_outputs.getArray()[i * batch_size + j])
            new_batch = Matrix.Matrix("",batch_size,data_length,mat)
            batches_out.append(new_batch)

        #starting multithreaded server
        print("Registering processes: [",end="")
        for i in range(cores_used):
            print(' %d ' % i,end="",flush=True)
            processes.append(Process(target=self.mtrain, args=(batches_in[i],batches_out[i],iterations,lock,i)))
        print("]")

        print("Starting processes: [",end="")
        j = 0
        for process in processes:
            print(' %d ' % j,end="",flush=True)
            process.start()
            j += 1
        print("]")

        print("Finishing processes: [",end="")
        j = 0
        for process in processes:
            print(' %d ' % j,end="",flush=True)
            process.join()
            j += 1
        print("]\n")
        print("Training is done")

        #combine batches
        self.synaptic_weights = self.synaptic_batches[0]

        '''for i in range(cores_used):
            self.synaptic_batches[i].printMatrix()
            print("")'''

        for i in range(cores_used - 1):
            self.synaptic_weights = self.synaptic_weights - self.synaptic_batches[i+1]