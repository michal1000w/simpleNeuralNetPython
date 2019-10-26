import random
import Matrix
from Import import Export

from multiprocessing import Process, Manager, Lock, Array, Pool, freeze_support
from multiprocessing.managers import BaseManager
from itertools import product, repeat
import os
import time

class NeuralNetwork:
    def __init__(self,neuron_inputs:int,neuron_count:int,seed:int,threads:int,force:bool,test_inputs:[],test_outputs:[]):
        random.seed(seed)
        self.SEED = seed
        self.neuron_count = neuron_count
        self.neuron_inputs = neuron_inputs
        self.threads = threads
        self.result_list = []
        self.force = force

        self.test_inputs = test_inputs
        self.test_outputs = test_outputs

        self.synaptic_weights = Matrix.Matrix("")
        self.wynik = Matrix.Matrix("")
        self.nazwy = []
        self.name = ""

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
                if (i == neuron_inputs):
                    weights += ' '
                else:
                    weights += ','
            weights += ']'

        self.synaptic_weights.add(weights)
        self.synaptic_weights = self.synaptic_weights.T()

        self.synaptic_weights_single = Matrix.Matrix(self.synaptic_weights.getString())

    def load_synaptic_weights(self,weights:Matrix.Matrix):
        self.synaptic_weights = weights
    
    def print_synaptic_weights(self):
        self.synaptic_weights.printMatrix()

    def set_name(self,name:str):
        self.name = name
        print("[NeuNet]  Name set:",name)

    def print_classified(self):
        klasy = self.neuron_count

        for i in range(klasy):
            if (self.wynik.getArray()[0][i] >= 0.5):
                print("[",self.nazwy[i],"]", end="")
        print("")

    def print_classified_new(self):
        klasy = self.neuron_count
        max_val = max(self.wynik.getArray()[0])
        max_pos = 0

        for i in range(klasy):
            if (self.wynik.getArray()[0][i] == max_val):
                max_pos = i
                break
        print("[",self.nazwy[max_pos],"] z prawdopodobieństwem:",max_val)

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

        print("[Single Threaded training]")
        from msvcrt import getch, kbhit
        print("Press any key to end training")

        print(" [ ",end="")

        for i in range(iterations):
            if (i%modulo == 0):
                print(str((i*100)/iterations)+"% ",end="",flush=True)
            
            #algorytm start
            output = self.nthink(training_inputs)
            error = training_outputs - output

            adjustment = training_inputs.T() * (output.sigmoid_derivative() ** error)

            self.synaptic_weights_single += adjustment

            if (kbhit()): #przerwanie
                    print(" [przerwanie] ", end="", flush=True)
                    break
        
        #self.synaptic_weights = self.synaptic_weights / 8
        print(" 100% ]")
        self.synaptic_weights = self.synaptic_weights_single

    def think(self,inputs:Matrix.Matrix,realOutput = Matrix.Matrix("[]")):
        self.wynik = (inputs * self.synaptic_weights).sigmoid() #oblicza spodziewany wynik

        #obliczanie funkcji loss
        if (not(realOutput.getString() == "" or realOutput.getString() == "[]")):
            self.loss = (realOutput - self.wynik).square().mean()
            print("Loss: ", self.loss)

        return self.wynik

    def nthink(self,inputs:Matrix.Matrix):
        self.wynik = (inputs * self.synaptic_weights_single).sigmoid()
        return self.wynik

    def test_loss(self,test_inputs:[],test_outputs:[]):
        data_count = len(test_inputs)
        wyniki = []

        #obliczanie wyników dla danych testowych
        for i in range(data_count):
            wyniki.append(self.think(test_inputs[i],Matrix.Matrix("[]")))

        #obliczanie sumy funkcji loss
        suma = 0.0
        for i in range(data_count):
            suma += (test_outputs[i].T() - wyniki[i]).square().mean()

        mianownik = float(data_count)
        
        return suma / mianownik

    #multithreading

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
            print("CORE: ",ID)
        finally:
            lock.release()


    def log_result(self,result):
        self.result_list.append(result)
    def mtrainpool(self,training_inputs:Matrix.Matrix,training_outputs:Matrix.Matrix,iterations:int,ID:int):
        output = Matrix.Matrix("")
        error = Matrix.Matrix("")
        adjustment = Matrix.Matrix("")

        synaptic_weights = self.average_weight[0]

   
        for i in range(iterations):
            #algorytm start
            output = (training_inputs * synaptic_weights).sigmoid()
            error = training_outputs - output

            adjustment = training_inputs.T() * (output.sigmoid_derivative() ** error)

            synaptic_weights += adjustment

        self.synaptic_batches[ID] = synaptic_weights
        return 1

    def automatic_thread_count(self,data_count:int):
        cpu_cores = int(os.cpu_count())
        output = 0
        mnoznik = 90

        for i in range(data_count):
            if (i % int(mnoznik) == 0):
                output += 1

        if (output > cpu_cores):
            output = cpu_cores

        return output

    def train_server(self,training_inputs:Matrix.Matrix,training_outputs:Matrix.Matrix,iterations:int):
        processes = []
        lock = Lock()

        #dzielenie na batche
        batches_in = []
        batches_out = []
        data_count = int(len(training_inputs.getArray()))
        if (self.threads <= os.cpu_count() and self.threads > 0):
            cpu_count = int(self.threads)
        else:
            cpu_count = self.automatic_thread_count(data_count)
        if (self.threads > 0 and self.force == True):
            cpu_count = int(self.threads)

        #przechodzenie na tryb jednordzeniowy jeżeli mało danych
        if (cpu_count == 1):
            self.train(training_inputs,training_outputs,iterations)
            print("Training is done")

            ex = Export(self.name)
            ex.save_weights(self.synaptic_weights)
        else:
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




            
            ''' #Create batches old
            data_length = len(training_inputs.getArray()[0])
            for i in range(cores_used): #batch_size * 
                mat = []
                for j in range(batch_size): #wiersze
                    mat.append(training_inputs.getArray()[i * batch_size + j])
                new_batch = Matrix.Matrix("",batch_size,data_length,mat)
                batches_in.append(new_batch)
                
            data_length = len(training_outputs.getArray()[0])
            for i in range(cores_used): #batch_size * 
                mat = []
                for j in range(batch_size): #wiersze
                    mat.append(training_outputs.getArray()[i * batch_size + j])
                new_batch = Matrix.Matrix("",batch_size,data_length,mat)
                batches_out.append(new_batch)'''
            
            #create batches [input data]
            data_length = int(len(training_inputs.getArray()[0]))
            left = left_size
            for i in range(int(cores_used)): #batch_size * 
                mat = []
                dod = int(left_size - left)
                for j in range(batch_size): #wiersze
                    mat.append(training_inputs.getArray()[i * batch_size + j + dod])
                if (left > 0):
                    mat.append(training_inputs.getArray()[i * batch_size + batch_size + dod])
                    left = left - 1
                    new_batch = Matrix.Matrix("",batch_size + 1,data_length,mat)
                else:
                    new_batch = Matrix.Matrix("",batch_size,data_length,mat)
                batches_in.append(new_batch)

            #create batches [output data]
            data_length = int(len(training_outputs.getArray()[0]))
            left = left_size
            for i in range(int(cores_used)):
                mat = []
                dod = int(left_size - left)
                for j in range(batch_size): #wiersze
                    mat.append(training_outputs.getArray()[i * batch_size + j + dod])
                if (left > 0):
                    mat.append(training_outputs.getArray()[i * batch_size + batch_size + dod])
                    left = left - 1
                    new_batch = Matrix.Matrix("",batch_size + 1,data_length,mat)
                else:
                    new_batch = Matrix.Matrix("",batch_size,data_length,mat)
                batches_out.append(new_batch)




            #starting multithreaded server
            #os.environ["OPENBLAS_MAIN_FREE"] = "1" #disable BLAS multithreading

            Iter = 100
            string = Matrix.Matrix("",self.neuron_inputs,self.neuron_count,None).getString()
            modulo = 5 * ((iterations / 100)/100)

            from msvcrt import getch, kbhit

            #new multithreaded Loop
            print("Press any key to end training")
            print("[ ",end="")
            freeze_support()
            loss_value = 0.0
            for j in range(int(iterations/100)):
                #wypisanie postępu
                if (j%modulo == 0):
                    print(str(round((j*100)/(iterations/100),0))+"% ",end="",flush=True)
                    loss_value = self.test_loss(self.test_inputs,self.test_outputs)
                    print("Loss: ",loss_value,flush=True)
                weights = Matrix.Matrix(string)

                pool = Pool(cores_used)
                self.result_list = []
                for i in range(cores_used):
                    #result.append(pool.apply_async(self.mtrainpool,(batches_in,batches_out,Iter,i)))
                    pool.apply_async(self.mtrainpool, args = (batches_in[i],batches_out[i],Iter,i), callback = self.log_result)
                pool.close()
                pool.join()

                if (kbhit()): #przerwanie
                    print(" [przerwanie] ", end="", flush=True)
                    break

                #combine batches
                for i in range(cores_used):
                    weights += self.synaptic_batches[i]
                weights = weights/cores_used
                self.average_weight[0] = weights

                #for testing
                self.synaptic_weights = self.average_weight[0]


            #Starting Loop OLD
            '''print("[ ",end="")
            for j in range(int(iterations/100)):
                #wypisanie postępu
                if (j%modulo == 0):
                    print(str(round((j*100)/(iterations/100),0))+"% ",end="",flush=True)

                weights = Matrix.Matrix(string)
                processes = []

                for i in range(cores_used):
                    processes.append(Process(target=self.mtrain, args=(batches_in[i],batches_out[i],Iter,lock,i,cores_used)))
                for process in processes:
                    process.start()

                if (kbhit()): #przerwanie
                    print(" [przerwanie] ", end="", flush=True)
                    for process in processes:
                        process.join()
                    break

                for process in processes:
                    process.join()
                #combine batches
                for i in range(cores_used):
                    weights += self.synaptic_batches[i]
                weights = weights/cores_used
                self.average_weight[0] = weights'''

            print(" 100% ]")
            self.synaptic_weights = self.average_weight[0]

            print("Training is done")

            ex = Export(self.name)
            ex.save_weights(self.synaptic_weights)

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
            