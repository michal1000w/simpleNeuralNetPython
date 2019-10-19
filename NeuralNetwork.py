import random
import Matrix

from multiprocessing import Process, Manager, Lock, Array
from multiprocessing.managers import BaseManager
import os
import time

class NeuralNetwork:
    def __init__(self,neuron_inputs:int,neuron_count:int,seed:int,threads:int):
        random.seed(seed)
        self.SEED = seed
        self.neuron_count = neuron_count
        self.neuron_inputs = neuron_inputs
        self.threads = threads

        self.synaptic_weights = Matrix.Matrix("")
        self.wynik = Matrix.Matrix("")
        self.nazwy = []

        #multithreaded
        self.synaptic_batches = []
        self.average_weight = []

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


    def mtrain(self,training_inputs:Matrix.Matrix,training_outputs:Matrix.Matrix,iterations:int,lock:Lock,ID:int,cores_used:int):
        output = Matrix.Matrix("")
        error = Matrix.Matrix("")
        adjustment = Matrix.Matrix("")

        #synaptic_weights = Matrix.Matrix("")
        lock.acquire()
        try:
            synaptic_weights = self.average_weight[0]
        finally:
            lock.release()

        #time.sleep(0.005*ID) #async
   
        for i in range(iterations):
            #algorytm start
            output = (training_inputs * synaptic_weights).sigmoid()
            error = training_outputs - output

            adjustment = training_inputs.T() * (output.sigmoid_derivative() ** error)

            synaptic_weights += adjustment

        lock.acquire()
        try:
            self.synaptic_batches[ID] = synaptic_weights
            #self.error_batches[ID] = adjustment
        finally:
            lock.release()

    def train_server(self,training_inputs:Matrix.Matrix,training_outputs:Matrix.Matrix,iterations:int):
        processes = []
        lock = Lock()

        #dzielenie na batche
        batches_in = []
        batches_out = []
        data_count = len(training_inputs.getArray())
        if (self.threads <= os.cpu_count() and self.threads > 0):
            cpu_count = self.threads
        else:
            cpu_count = os.cpu_count()

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

        self.average_weight = manager.list()
        self.average_weight.append(self.synaptic_weights)




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
        Iter = 100
        string = Matrix.Matrix("",self.neuron_inputs,self.neuron_count,None).getString()
        modulo = 5 * ((iterations / 100)/100)

        print("[ ",end="")
        for j in range(int(iterations/100)):
            #wypisanie postępu
            if (j%modulo == 0):
                print(str(round((j*100)/(iterations/100),0))+"% ",end="",flush=True)

            processes = []
            weights = Matrix.Matrix(string)
            for i in range(cores_used):
                processes.append(Process(target=self.mtrain, args=(batches_in[i],batches_out[i],Iter,lock,i,cores_used)))
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            #combine batches
            for i in range(cores_used):
                weights += self.synaptic_batches[i]
            weights = weights/cores_used
            self.average_weight[0] = weights

        print(" 100% ]")
        self.synaptic_weights = self.average_weight[0]

        print("Training is done")

        '''
        print("Registering processes: [",end="")
        for i in range(cores_used):
            print(' %d ' % i,end="",flush=True)
            processes.append(Process(target=self.mtrain, args=(batches_in[i],batches_out[i],Iter,lock,i,cores_used)))
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
        print("]\n")'''
        