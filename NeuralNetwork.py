import random
import Matrix
from Import import Export

from multiprocessing import Process, Manager, Lock, Array, Pool, freeze_support
from multiprocessing.managers import BaseManager
from itertools import product, repeat
import os
import time

class NeuralNetwork:
    def __init__(self,neuron_inputs:int,neuron_count:int,seed:int,threads:int,force:bool,test_inputs:[],test_outputs:[],hidden_Layout:Matrix.Matrix):
        random.seed(seed)
        self.SEED = seed
        self.neuron_count = []
        self.neuron_inputs = []

        #self.neuron_count.append(neuron_count)
        #self.neuron_inputs.append(neuron_inputs)
        self.threads = threads
        self.result_list = []
        self.force = force

        self.test_inputs = test_inputs
        self.test_outputs = test_outputs

        self.synaptic_weights = Matrix.Matrix("")
        self.synaptic_weights_single = Matrix.Matrix("")
        self.wynik = Matrix.Matrix("")
        self.nazwy = []
        self.name = ""

        #multithreaded
        manager = Manager()
        self.synaptic_batches = []
        self.all_synaptic_batches = manager.list()
        self.average_weight = []
        #deep learning
        self.hidden_Layout = hidden_Layout
        self.all_layer_weights = []
        self.all_layer_weights_single = []
        self.parameter_b = []

        self.layers_count = len(self.hidden_Layout.getArray())
        self.layers_count_hidden = 0
        
        if (self.layers_count > 1):
            self.read_hidden_Layout(neuron_inputs,neuron_count)
        self.generate_synaptic_weights()
        self.generate_parameter_b()

    def read_hidden_Layout(self,neuron_inputs:int,neuron_count:int):
        self.layers_count_hidden = 0
        if (self.layers_count > 1):
            self.layers_count_hidden = self.layers_count - 2

        hiddenL = self.hidden_Layout.getArray()
        if (self.layers_count > 1):
            hiddenL[0] = [neuron_inputs]
            hiddenL[self.layers_count-1] = [neuron_count]
        

        for i in range(1 , self.layers_count):
            self.neuron_inputs.append(int(hiddenL[i][0]))
            self.neuron_count.append(int(hiddenL[i-1][0]))

        print("Hidden layout: ")
        for i in range(self.layers_count - 1):
            print(self.neuron_inputs[i],self.neuron_count[i])

    def generate_synaptic_weights(self):
        self.all_layer_weights = []
        self.all_layer_weights_single = []

        for z in range(self.layers_count - 1):
            weights = ""
            liczba = ""
            for j in range(self.neuron_count[z]):
                weights += '['
                for i in range(self.neuron_inputs[z]):
                    liczba = str((random.random() * 2) - 1)
                    for y in range(len(liczba)):
                        weights += liczba[y]
                    if (i == self.neuron_inputs[z]-1):
                        weights += ' '
                    else:
                        weights += ','
                weights += ']'

            self.synaptic_weights = Matrix.Matrix("")
            self.synaptic_weights.add(weights)
            #self.synaptic_weights = self.synaptic_weights.T()

            self.synaptic_weights_single = Matrix.Matrix(self.synaptic_weights.getString())


            self.all_layer_weights.append(self.synaptic_weights)
            self.all_layer_weights_single.append(self.synaptic_weights_single)

    def generate_parameter_b(self):
        for z in range(self.layers_count - 1):
            weights = ""
            liczba = ""
            for j in range(len(self.all_layer_weights[z].getArray()[0])):
                weights += "[ 0 ]"

            self.parameter_b.append(Matrix.Matrix(weights))

    def load_synaptic_weights(self,weights:Matrix.Matrix): ###TO DO
        self.synaptic_weights = weights
    
    def print_parameter_b(self):
        print("\nParameter b: \n")
        for z in range(self.layers_count - 1):
            print("Layer [",z+1,"]\n")
            self.parameter_b[z].printMatrix()
            print("")


    def print_synaptic_weights(self):
        print("")
        print("All layers count: ",self.layers_count - 1)
        print("Hidden layers count: ",self.layers_count_hidden)
        print("")
        for z in range(self.layers_count - 1):
            print("Layer [",z+1,"]\n")
            self.all_layer_weights[z].printMatrix()
            print("")

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
        ####TO DO (almost done)
        output = Matrix.Matrix("")
        error = Matrix.Matrix("")
        adjustment = Matrix.Matrix("")

        modulo = 5 * (iterations / 100)

        #deep learning
        synaptic_weights = []
        layers_number = self.layers_count - 1
        for z in range(layers_number):
            synaptic_weights.append(self.all_layer_weights_single[z])

        print("[Single Threaded training]")
        from msvcrt import getch, kbhit
        print("Press any key to end training\n\n")

        print(" [ ",end="")


        for i in range(iterations):
            if (i%modulo == 0):
                print(str((i*100)/iterations)+"% ",end="",flush=True)
                loss_value = self.test_loss(self.test_inputs,self.test_outputs)
                print("Loss: ",loss_value,flush=True)
        
            from msvcrt import getch
            #algorytm start #1
            wyniki_czastkowe = []
            output = training_inputs
            for z in range(layers_number):
                output = (output * self.all_layer_weights[z]).sigmoid() #oblicza spodziewany wynik
                wyniki_czastkowe.append(output)


            delta = [] #1

            #layer ostatni
            error = training_outputs - output  #różnica między prawdziwym, a przewidywanym wynikiem
            delta.append(error ** output.sigmoid_derivative())
            #1


            ilosc_do_przeliczenia = layers_number - 1 #jedna mniej


            #kolejne layery
            
            for z in range(0,ilosc_do_przeliczenia): 
                error = delta[z] * synaptic_weights[ilosc_do_przeliczenia - z].T()  #działa dla 1 hidden
                delta.append(error ** wyniki_czastkowe[ilosc_do_przeliczenia - z - 1].sigmoid_derivative())


            #calculate adjustments
            
            synaptic_weights[0] += (training_inputs.T() * delta[layers_number-1])

            j = 1
            for z in reversed(range(0,layers_number - 1)):
                synaptic_weights[j] += wyniki_czastkowe[j-1].T() * delta[z]
                j += 1

            
            for z in range(layers_number): #zapis wag do zmiennej globalnej
                self.all_layer_weights_single[z] = synaptic_weights[z]
            
            self.all_layer_weights = self.all_layer_weights_single

            if (kbhit()): #przerwanie
                    print(" [przerwanie] ", end="", flush=True)
                    break
        
        print(" 100% ]")
        self.all_layer_weights = self.all_layer_weights_single

    def think(self,inputs:Matrix.Matrix,realOutput = Matrix.Matrix("[]")):
        self.wynik = inputs
        for z in range( self.layers_count_hidden + 1):
            self.wynik = (self.wynik * self.all_layer_weights[z]).sigmoid() #oblicza spodziewany wynik
            
            
        #obliczanie funkcji loss
        if (not(realOutput.getString() == "" or realOutput.getString() == "[]")):
            self.loss = (realOutput - self.wynik).square().mean()
            print("Loss: ", self.loss)

        return self.wynik

    def nthink(self,inputs:Matrix.Matrix):
        self.wynik = inputs
        for z in range(len(self.hidden_Layout.getArray())):
            self.wynik = (self.wynik * self.all_layer_weights_single[z]).sigmoid()
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
            suma += (test_outputs[i] - wyniki[i]).square().mean()

        mianownik = float(data_count)
        
        return suma / mianownik

    def test_training(self,test_inputs:[],test_outputs:[]):
        data_count = len(test_inputs)
        wyniki = []
        #obliczanie wyników dla danych testowych
        for i in range(data_count):
            wyniki.append(self.think(test_inputs[i],Matrix.Matrix("[]")))

        return wyniki



    #multithreading
    def create_loss_weights(self,test_inputs:[],test_outputs:[],synaptic_weights:[],layers:int,ID:int):
        data_count = len(test_inputs)
        wyniki = []

        #obliczanie wyników dla danych testowych
        for i in range(data_count):
            wyniki.append(self.lthink(test_inputs[i],synaptic_weights,layers,ID))

        #obliczanie sumy funkcji loss
        suma = 0.0
        for i in range(data_count):
            suma += (test_outputs[i] - wyniki[i]).square().mean()

        mianownik = float(data_count)
        
        return suma / mianownik

    def lthink(self,inputs:Matrix.Matrix,synaptic_weights:[],layers:int,ID:int):
        self.wynik = inputs
        for z in range(self.layers_count - 1):
            self.wynik = (self.wynik * synaptic_weights[ID * layers + z]).sigmoid()
        return self.wynik


    def backward(self,training_inputs:Matrix.Matrix,training_outputs:Matrix.Matrix,iterations:int,ID:int,layers:int):
        output = Matrix.Matrix("")
        error = Matrix.Matrix("")
        adjustment = Matrix.Matrix("")
        #synaptic_weights = Matrix.Matrix("")
        synaptic_weights = []
        layers_number = self.layers_count - 1

        for z in range(layers_number):
            synaptic_weights.append(self.average_weight[z])

    
        for i in range(iterations):
            wyniki_czastkowe = []
            output = training_inputs
            for z in range(layers_number):
                output = (output * synaptic_weights[z]).sigmoid()
                wyniki_czastkowe.append(output)

            delta = []

            error = training_outputs - output #1
            delta.append(error ** output.sigmoid_derivative()) #1

            ilosc_do_przeliczenia = layers_number - 1

            for z in range(0, ilosc_do_przeliczenia): #1
                error = delta[z] * synaptic_weights[ilosc_do_przeliczenia - z].T()
                delta.append(error ** wyniki_czastkowe[ilosc_do_przeliczenia - z - 1].sigmoid_derivative())

            
            synaptic_weights[0] += (training_inputs.T() * delta[layers_number-1]) #1
            
            j = 1
            for z in reversed(range(0,layers_number - 1)):
                synaptic_weights[j] += (wyniki_czastkowe[j-1].T() * delta[z])
                j += 1

    
        for z in range(layers_number):
            self.all_synaptic_batches[ID * layers_number + z] = synaptic_weights[z]
            
        return 1


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
            self.all_synaptic_batches = manager.list()
            
            for i in range(self.layers_count - 1):
                self.synaptic_batches.append(self.all_layer_weights[i])

            
            self.average_weight = manager.list()
            for i in range(self.layers_count - 1):
                self.average_weight.append(Matrix.Matrix(self.all_layer_weights[i].getString()))

            '''for i in range(cores_used):
                self.all_synaptic_batches.append(self.average_weight)'''

            for i in range(cores_used * (self.layers_count - 1)):
                self.all_synaptic_batches.append(Matrix.Matrix(self.average_weight[i % (self.layers_count - 1)].getString()))

            


            
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
            hidden_Layout_count = self.layers_count - 1
            string = []
            for i in range(hidden_Layout_count):
                string.append(Matrix.Matrix("",self.neuron_inputs[i],self.neuron_count[i],None).T().getString())
            
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

                weights = []
                for i in range(hidden_Layout_count):
                    weights.append(Matrix.Matrix(string[i]))

                pool = Pool(cores_used)
                for i in range(cores_used):
                    #result.append(pool.apply_async(self.mtrainpool,(batches_in,batches_out,Iter,i)))
                    result = pool.apply_async(self.backward, args = (batches_in[i],batches_out[i],Iter,i,hidden_Layout_count), callback = self.log_result)
                    
                pool.close()
                pool.join()


                if (kbhit()): #przerwanie
                    print(" [przerwanie] ", end="", flush=True)
                    break

                #combine batches OLD
                '''for z in range(hidden_Layout_count):
                    self.average_weight[z] = self.all_synaptic_batches[z][0]'''
                    

                '''for z in range(hidden_Layout_count): 
                    for i in range(cores_used):
                        weights[z] += self.all_synaptic_batches[z][i] #tu błąd

                    self.all_synaptic_batches[z][0].printMatrix()
                    print("\n\n")
                    self.all_synaptic_batches[z][1].printMatrix()
                    break'''
                    #weights[z] = weights[z]/cores_used
                    #self.average_weight[z] = weights[z]

                #combine batches NEW to do in deeplearning
                #creating mean weights

                #print(len(self.all_synaptic_batches[0]))

                
                for z in range(hidden_Layout_count): #chyba dobre
                    mean_weights = []
                    mean_weights_sum = 0.0

                    for i in range(cores_used):
                        #self.all_synaptic_batches[i][z].printMatrix()
                        mean_weights.append(1 / self.create_loss_weights(self.test_inputs,self.test_outputs,self.all_synaptic_batches,hidden_Layout_count,i))
                        #weights[z].printSize()
                        weights[z] += (self.all_synaptic_batches[i * hidden_Layout_count + z] % mean_weights[i]) ##tutaj sprawdzić

                        mean_weights_sum += mean_weights[i]
                        

                    weights[z] = weights[z]/mean_weights_sum
                    self.average_weight[z] = weights[z]
                '''
                for z in range(hidden_Layout_count):
                    self.average_weight[z] = self.all_synaptic_batches[z]
                '''
                '''
                    self.average_weight[z].printMatrix()
                    print("\n")
                '''
                '''mean_weights = []
                mean_weights_sum = 0.0
                for i in range(cores_used):
                    mean_weights.append(1 / self.create_loss_weights(self.test_inputs,self.test_outputs,self.synaptic_batches[i]))
                    #print(mean_weights)
                    weights += self.synaptic_batches[i] % mean_weights[i]
                    mean_weights_sum += mean_weights[i]
                weights = weights/mean_weights_sum
                self.average_weight[0] = weights'''
                


                #for testing
                for i in range(self.layers_count - 1):
                    self.all_layer_weights[i] = self.average_weight[i]

                #print(self.average_weight[0])


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
            for i in range(self.layers_count - 1):
                self.all_layer_weights[i] = self.average_weight[i]
                self.synaptic_weights = self.average_weight[i]

            print("Training is done")

            ex = Export(self.name) ############TO DO

            ex.save_weights(self.all_layer_weights[0])

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
            