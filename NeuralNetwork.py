import random
import Matrix
from Import import Export

from multiprocessing import Process, Manager, Lock, Array, Pool, freeze_support
from multiprocessing.managers import BaseManager
from itertools import product, repeat
import os
import time
from colorama import Fore, Back, Style
#CUDA
import cupy as cp

class NeuralNetwork:
    def __init__(self,neuron_inputs:int,neuron_count:int,seed:int,threads:int,force:bool,test_inputs:[],test_outputs:[],hidden_Layout:Matrix.Matrix,print_synaptic = True):
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

        self.device = 0

        #multithreaded
        manager = Manager()
        self.synaptic_batches = []
        self.all_synaptic_batches = manager.list()
        self.average_weight = []
        self.iter = 100
        #deep learning
        self.hidden_Layout = hidden_Layout
        self.all_layer_weights = []
        self.all_layer_weights_single = []
        self.parameter_b = []
        self.print_synaptic = print_synaptic
        #GPU
        self.all_synaptic_weights = manager.list()
        self.average_weight_cuda = []
        self.mean_weights_multi = manager.list()

        self.arythmetic_mean = False
        self.float_reduction = False

        #deep learning
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

        print("Layers count: ",self.layers_count - 1)
        if (self.print_synaptic):
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
            self.all_synaptic_weights.append(self.Matrix_to_cupy_single(self.synaptic_weights))

    def print_synaptic_set(self,set:bool):
        self.print_synaptic = set

    def force_arythmetic_mean(self,force):
        self.arythmetic_mean = force

    def force_float_reduction(self,force):
        self.float_reduction = force

    def generate_parameter_b(self):
        for z in range(self.layers_count - 1):
            weights = ""
            liczba = ""
            for j in range(len(self.all_layer_weights[z].getArray()[0])):
                weights += "[ 0 ]"

            self.parameter_b.append(Matrix.Matrix(weights))

    def load_synaptic_weights(self,weights:[]):
        self.all_layer_weights = weights
        for i in range(len(weights)):
            self.all_synaptic_weights.append(self.Matrix_to_cupy_single(weights[i]))
    
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
        print(Fore.CYAN + "[NeuNet]" + Fore.RESET + "  Name set:",name)

    def set_iter(self,iter:int):
        self.iter = iter

    def set_device(self,dev:int):
        self.device = dev

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
        #zmienne
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
                loss_value, skutecznosc = self.test_loss_extended(self.test_inputs,self.test_outputs)
                print("Loss: ",loss_value," Skutecznosc: ",skutecznosc,"%",flush=True)
        
            
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
            
            synaptic_weights[0] += (training_inputs.T() * delta[ilosc_do_przeliczenia])

            j = 1
            for z in reversed(range(ilosc_do_przeliczenia)):
                synaptic_weights[j] += wyniki_czastkowe[j-1].T() * delta[z]
                j += 1

            
            for z in range(layers_number): #zapis wag do zmiennej globalnej
                self.all_layer_weights_single[z] = synaptic_weights[z]
            
            self.all_layer_weights = self.all_layer_weights_single

            if (kbhit()): #przerwanie
                print(Fore.RED + " [przerwanie] " + Fore.RESET, end="", flush=True)
                break
        
        print(" 100% ]")
        self.all_layer_weights = self.all_layer_weights_single

    def think(self,inputs:Matrix.Matrix,realOutput = Matrix.Matrix("[]")):
        self.wynik = inputs
        for z in range( self.layers_count - 1):
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

    def test_loss_extended(self,test_inputs:[],test_outputs:[]):
        data_count = len(test_inputs)
        wyniki = []
        dobre = 0.0

        #obliczanie wyników dla danych testowych
        output_count = len(test_outputs[0].getArray()[0])
        for i in range(data_count):
            wyniki.append(self.think(test_inputs[i],Matrix.Matrix("[]")))

            max_val = max(wyniki[i].getArray()[0])
            for j in range(output_count):
                if (wyniki[i].getArray()[0][j] == max_val):
                    if (test_outputs[i].getArray()[0][j] == 1):
                        dobre += 1
                    break
        procent_poprawnych = (dobre * 100.0) / float(len(wyniki))

        #obliczanie sumy funkcji loss
        suma = 0.0
        for i in range(data_count):
            suma += (test_outputs[i] - wyniki[i]).square().mean()

        mianownik = float(data_count)
        
        return suma / mianownik, procent_poprawnych

    def test_training(self,test_inputs:[],test_outputs:[]):
        data_count = len(test_inputs)
        wyniki = []
        #obliczanie wyników dla danych testowych
        for i in range(data_count):
            wyniki.append(self.think(test_inputs[i]))

        return wyniki

    def test_training_cuda(self,test_inputs:[],test_outputs:[]):
        data_count = len(test_inputs)
        wyniki = []
        syn_weights,t_inputs = self.Matrix_to_cupy_test(test_inputs,self.all_layer_weights)
        sigmoid = cp.ElementwiseKernel(
                'float32 in',
                'float32 out',
                '''
                float h = exp(-1 * in);
                out = 1 / (1 + h);
                ''',
                'sigmoid'
            )

        for i in range(data_count):
            wyniki.append(self.think_CUDA(t_inputs[i],syn_weights,sigmoid))

        return self.Convert_Output(wyniki)


    #CUDA compute
    def Matrix_to_cupy_single(self,inputs:Matrix.Matrix):
        wiersze = len(inputs.getArray())
        kolumny = len(inputs.getArray()[0])

        if (self.float_reduction):
            output = cp.arange(wiersze*kolumny,dtype=cp.float64).reshape(wiersze,kolumny)
        else:
            output = cp.arange(wiersze*kolumny,dtype=cp.float32).reshape(wiersze,kolumny)
        for i in range(wiersze):
            for j in range(kolumny):
                output[i][j] = inputs.getArray()[i][j]

        return output


    def Matrix_to_cupy(self,train_inputs:Matrix.Matrix,train_outputs:Matrix.Matrix,synaptic_weights:[]):
        #time
        start = time.perf_counter()
        print(Fore.LIGHTBLUE_EX + "Copying data to GPU..." + Fore.RESET)
        
        #inputs to cp
        wiersze = len(train_inputs.getArray())
        kolumny = len(train_inputs.getArray()[0])

        if (self.float_reduction):
            inputs = cp.arange(wiersze*kolumny,dtype=cp.float64).reshape(wiersze,kolumny)
        else:
            inputs = cp.arange(wiersze*kolumny,dtype=cp.float32).reshape(wiersze,kolumny)
        for i in range(wiersze):
            for j in range(kolumny):
                inputs[i][j] = train_inputs.getArray()[i][j]

        #outputs to cp
        wiersze = len(train_outputs.getArray())
        kolumny = len(train_outputs.getArray()[0])

        if (self.float_reduction):
            outputs = cp.arange(wiersze*kolumny,dtype=cp.float64).reshape(wiersze,kolumny)
        else:
            outputs = cp.arange(wiersze*kolumny,dtype=cp.float32).reshape(wiersze,kolumny)
        for i in range(wiersze):
            for j in range(kolumny):
                outputs[i][j] = train_outputs.getArray()[i][j]

        #synaptic_weights to cp
        syn_weights = []
        layers = len(synaptic_weights)
        for z in range(layers):
            wiersze = len(synaptic_weights[z].getArray())
            kolumny = len(synaptic_weights[z].getArray()[0])

            if (self.float_reduction):
                weights = cp.arange(wiersze*kolumny,dtype=cp.float64).reshape(wiersze,kolumny)
            else:
                weights = cp.arange(wiersze*kolumny,dtype=cp.float32).reshape(wiersze,kolumny)
            for i in range(wiersze):
                for j in range(kolumny):
                    weights[i][j] = synaptic_weights[z].getArray()[i][j]
            syn_weights.append(weights)

        #test_inputs
        t_inputs = []
        length = len(self.test_inputs)
        for z in range(length):
            wiersze = len(self.test_inputs[z].getArray())
            kolumny = len(self.test_inputs[z].getArray()[0])

            if (self.float_reduction):
                mat = cp.arange(wiersze*kolumny,dtype=cp.float64).reshape(wiersze,kolumny)
            else:
                mat = cp.arange(wiersze*kolumny,dtype=cp.float32).reshape(wiersze,kolumny)
            for i in range(wiersze):
                for j in range(kolumny):
                    mat[i][j] = self.test_inputs[z].getArray()[i][j]
            t_inputs.append(mat)

        #test_outputs
        t_outputs = []
        length = len(self.test_outputs)
        for z in range(length):
            wiersze = len(self.test_outputs[z].getArray())
            kolumny = len(self.test_outputs[z].getArray()[0])

            if (self.float_reduction):
                mat = cp.arange(wiersze*kolumny,dtype=cp.float64).reshape(wiersze,kolumny)
            else:
                mat = cp.arange(wiersze*kolumny,dtype=cp.float32).reshape(wiersze,kolumny)
            for i in range(wiersze):
                for j in range(kolumny):
                    mat[i][j] = self.test_outputs[z].getArray()[i][j]
            t_outputs.append(mat)

        #time
        duration = time.perf_counter() - start
        print(Fore.GREEN + "Copying data done in:" + Fore.RESET,duration,"s")

        return (inputs,outputs,syn_weights,t_inputs,t_outputs)

    def Matrix_to_cupy_test(self,test_inputs,synaptic_weights):
        #synaptic_weights to cp
        syn_weights = []
        layers = len(synaptic_weights)
        for z in range(layers):
            wiersze = len(synaptic_weights[z].getArray())
            kolumny = len(synaptic_weights[z].getArray()[0])

            if (self.float_reduction):
                weights = cp.arange(wiersze*kolumny,dtype=cp.float64).reshape(wiersze,kolumny)
            else:
                weights = cp.arange(wiersze*kolumny,dtype=cp.float32).reshape(wiersze,kolumny)
            for i in range(wiersze):
                for j in range(kolumny):
                    weights[i][j] = synaptic_weights[z].getArray()[i][j]
            syn_weights.append(weights)

        #test_inputs
        t_inputs = []
        length = len(test_inputs)
        for z in range(length):
            wiersze = len(test_inputs[z].getArray())
            kolumny = len(test_inputs[z].getArray()[0])

            if (self.float_reduction):
                mat = cp.arange(wiersze*kolumny,dtype=cp.float64).reshape(wiersze,kolumny)
            else:
                mat = cp.arange(wiersze*kolumny,dtype=cp.float32).reshape(wiersze,kolumny)
            for i in range(wiersze):
                for j in range(kolumny):
                    mat[i][j] = test_inputs[z].getArray()[i][j]
            t_inputs.append(mat)

        return syn_weights, t_inputs

    def Convert_Weights(self,synaptic_weights):
        syn_weights = []

        for z in range(len(synaptic_weights)):
            wiersze = len(synaptic_weights[z])
            kolumny = len(synaptic_weights[z][0])
            mat = []
            for i in range(wiersze):
                rowList = []
                for j in range(kolumny):
                    rowList.append(synaptic_weights[z][i][j])
                mat.append(rowList)
            syn_weights.append(Matrix.Matrix("",wiersze,kolumny,mat))

        return syn_weights

    def Convert_Output(self,t_output:[]):
        output = []
        for z in range(len(t_output)):
            wiersze = len(t_output[z])
            kolumny = len(t_output[z][0])
            mat = []
            for i in range(wiersze):
                rowList = []
                for j in range(kolumny):
                    rowList.append(t_output[z][i][j])
                mat.append(rowList)
            output.append(Matrix.Matrix("",wiersze,kolumny,mat))
        
        return output

    def think_CUDA(self,inputs,synaptic_weights,sigmoid,layers = 1,ID = 0):
        output = inputs

        for z in range(self.layers_count - 1):
            output = sigmoid(output.dot(synaptic_weights[(ID * layers + z)]))

        return output

    def test_loss_eCuda(self,t_in:[],t_out:[],syn_weights,sigmoid):
        data_count = len(t_in)
        wyniki = []
        dobre = 0.0

        #obliczanie wyników testowych
        output_count = len(t_out[0][0])
        for i in range(data_count):
            wyniki.append(self.think_CUDA(t_in[i],syn_weights,sigmoid))

            max_val = max(wyniki[i][0])
            for j in range(output_count):
                if (wyniki[i][0][j] == max_val):
                    if (t_out[i][0][j] == 1):
                        dobre += 1
                    break
        skutecznosc = (dobre * 100.0) / float(len(wyniki))

        #obliczanie sumy funkcji loss
        suma = 0.0
        for i in range(data_count):
            suma += (cp.square(t_out[i] - wyniki[i])).mean(axis=None)

        mianownik = float(data_count)

        return ((suma/mianownik),skutecznosc)
        

    def CUDA_train(self,training_inputs:Matrix.Matrix,training_outputs:Matrix.Matrix,iterations:int):
        print("\n\nTraining device: CUDA")

        #converting data
        inputs, outputs, syn_weights, t_in, t_out = self.Matrix_to_cupy(training_inputs,training_outputs,self.all_layer_weights)
        
        print("Setting device to primary GPU")
        cp.cuda.Device(0).use()

        #Przygotowanie zmiennych
        modulo = 5 * (iterations/100)
        layers_number = self.layers_count - 1

        ###############   KERNELS   #################
        if (self.float_reduction):  
            sigmoid = cp.ElementwiseKernel(
                'float64 in',
                'float64 out',
                '''
                float h = exp(-1 * in);
                out = 1 / (1 + h);
                ''',
                'sigmoid'
            )
            sigmoid_derivative = cp.ElementwiseKernel(
                'float64 in',
                'float64 out',
                '''
                out = in * (1 - in);
                ''',
                'sigmoid_derivative'
            )
        else:
            sigmoid = cp.ElementwiseKernel(
                'float32 in',
                'float32 out',
                '''
                float h = exp(-1 * in);
                out = 1 / (1 + h);
                ''',
                'sigmoid'
            )
            sigmoid_derivative = cp.ElementwiseKernel(
                'float32 in',
                'float32 out',
                '''
                out = in * (1 - in);
                ''',
                'sigmoid_derivative'
            )
        #############################################

        #trening

        print("[GPU Training]")
        from msvcrt import getch, kbhit
        print("Press any key to end training\n\n")

        print(" [ ",end="")

        for i in range(iterations):
            if (i%modulo == 0):
                print(str((i*100)/iterations)+"% ",end="",flush=True)
                #To z jakiegoś powodu nie chce działać
                '''
                loss_value, skutecznosc = self.test_loss_eCuda(t_in,t_out,syn_weights,sigmoid)
                print("Loss: ",loss_value," Skutecznosc: ",skutecznosc,"%",flush=True)
                '''
            
            #algorytm start
            wyniki_czastkowe = []
            output = inputs
            for z in range(layers_number):
                output = sigmoid(output.dot(syn_weights[z]))
                wyniki_czastkowe.append(output)

            delta = []
            #1

            #layer ostatni
            error = outputs - output
            delta.append(error * sigmoid_derivative(output))


            ilosc_do_przeliczenia = layers_number - 1

            #kolejne layery
            for z in range(ilosc_do_przeliczenia):
                error = delta[z].dot(syn_weights[ilosc_do_przeliczenia - z].T)
                delta.append(error * sigmoid_derivative(wyniki_czastkowe[ilosc_do_przeliczenia - z - 1]))

            #calculate adjustments
            syn_weights[0] += (inputs.T).dot(delta[ilosc_do_przeliczenia])

            j = 1
            for z in reversed(range(ilosc_do_przeliczenia)):
                syn_weights[j] += (wyniki_czastkowe[j-1].T).dot(delta[z])
                j += 1



            if (kbhit()): #przerwanie
                print(Fore.RED + " [przerwanie] " + Fore.RESET,end="",flush=True)
                break


        print(" 100% ]")
        self.all_layer_weights = self.Convert_Weights(syn_weights)



    def CUDA_train_batch(self,training_inputs,training_outputs,iterations:int,ID,sigmoid,sigmoid_derivative,sigmDot,lock):
        layers_number = self.layers_count - 1
        synaptic_weights = []

        for z in range(layers_number):
            lock.acquire()
            synaptic_weights.append(self.average_weight_cuda[z])
            lock.release()
        
        

        for i in range(iterations):
            #algorytm start
            wyniki_czastkowe = []
            output = training_inputs
            for z in range(layers_number):
                output = sigmoid(output.dot(synaptic_weights[z]))
                
                ''' #Tutaj coś nie działa i są ogromne wyniki
                wiersze = cp.int32(output.size/output[0].size)
                kolumny = cp.int32(synaptic_weights[z][0].size)
                N = cp.int32(wiersze * kolumny)
                out = cp.arange(N,dtype=cp.float32).reshape(wiersze,kolumny)
                
                sigmDot((wiersze,),(kolumny,),(output,synaptic_weights[z],out),shared_mem = N)
                output = sigmoid(out)
                '''
                
                wyniki_czastkowe.append(output)

            
            delta = []
            

            #layer ostatni
            error = training_outputs - output
            delta.append(error * sigmoid_derivative(output))

            ilosc_do_przeliczenia = layers_number - 1

            #kolejna layery
            for z in range(ilosc_do_przeliczenia):
                error = delta[z].dot(synaptic_weights[ilosc_do_przeliczenia - z].T)
                delta.append(error * sigmoid_derivative(wyniki_czastkowe[ilosc_do_przeliczenia - z - 1]))

            #calculte adjustments
            synaptic_weights[0] += (training_inputs.T).dot(delta[ilosc_do_przeliczenia])

            j = 1
            for z in reversed(range(ilosc_do_przeliczenia)):
                synaptic_weights[j] += (wyniki_czastkowe[j-1].T).dot(delta[z])
                j += 1

        for z in range(layers_number):
            lock.acquire()
            self.all_synaptic_batches[ID * layers_number + z] = synaptic_weights[z]
            lock.release()
        
        return 1

    def CUDA_train_Server_Fast(self,training_inputs:Matrix.Matrix,training_outputs:Matrix.Matrix,iterations:int):
        print(Fore.LIGHTBLUE_EX + "Starting CUDA server..." + Fore.RESET)
        cp.cuda.Device(0).use()

        #################CUDA KERNELS######################
        if (self.float_reduction):  
            sigmoid = cp.ElementwiseKernel(
                'float64 in',
                'float64 out',
                '''
                float h = exp(-1 * in);
                out = 1 / (1 + h);
                ''',
                'sigmoid'
            )
            sigmoid_derivative = cp.ElementwiseKernel(
                'float64 in',
                'float64 out',
                '''
                out = in * (1 - in);
                ''',
                'sigmoid_derivative'
            )
        else:
            sigmoid = cp.ElementwiseKernel(
                'float32 in',
                'float32 out',
                '''
                float h = exp(-1 * in);
                out = 1 / (1 + h);
                ''',
                'sigmoid'
            )
            sigmoid_derivative = cp.ElementwiseKernel(
                'float32 in',
                'float32 out',
                '''
                out = in * (1 - in);
                ''',
                'sigmoid_derivative'
            )
            
            
            sigmDot = cp.RawKernel( #coś tutaj nie działa
                r'''
                extern "C" __global__
                void sigmDot(const float* a,const float* b, float* out){
                    extern __shared__ float array[];

                    const int n = blockDim.x * blockDim.y;
                    float* sh_data = (float*)array;
                    float* temp = (float*)&sh_data[n];
                    
                    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];

                    __syncthreads();

                    if (0 == threadIdx.x){
                        float sum = 0;
                        for (int i = 0; i < n; i++)
                            sum += temp[i];
                        *out = 1 / (1 + (exp(-1*sum)));
                    }
                }
                ''',
                'sigmDot'
            )
            
        ###################################################

        #Przygotowanie zmiennych
        batches_in = []
        batches_out = []
        data_count = int(len(training_inputs.getArray()))
        cpu_count = self.automatic_thread_count_gpu(data_count)
        all_inputs, all_outputs, all_synaptic_weights, t_in, t_out = self.Matrix_to_cupy(training_inputs,training_outputs,self.all_layer_weights)
        self.all_synaptic_weights = (all_synaptic_weights)


        #przechodzenie na tryb jednordzeniowy gdy mało danych
        if (cpu_count == 1):
            self.CUDA_train(training_inputs,training_outputs,iterations)
            print(Fore.GREEN + "Training is done" + Fore.RESET)
            ex = Export(self.name)
            ex.save_weights(self.all_layer_weights)
        else:
            #ustalenie ilości używanych wątków
            if (data_count >= cpu_count):
                batch_size = int(data_count / cpu_count)
                left_size = data_count - cpu_count*batch_size
                cores_used = cpu_count
            else:
                batch_size = 1
                left_size = 0
                cores_used = data_count

            print(Fore.LIGHTBLUE_EX + "\n[Creating Batches]" + Fore.RESET)
            #print("CPU Threads: ",os.cpu_count())
            print("Data count: ",data_count)
            print("Batch size: ",batch_size)
            print("Left size: ",left_size)
            print("Threads used: ",cores_used)
            print("")

            #pamięć współdzielona
            manager = Manager()
            self.synaptic_batches = manager.list()
            self.all_synaptic_batches = manager.list()

            for i in range(self.layers_count - 1):
                self.synaptic_batches.append(self.all_synaptic_weights[i])
            
            self.average_weight_cuda = manager.list()
            
            for i in range(self.layers_count - 1):
                self.average_weight_cuda.append(self.all_synaptic_weights[i])


            for i in range(cores_used * (self.layers_count - 1)):
                self.all_synaptic_batches.append(self.average_weight_cuda[i % (self.layers_count - 1)])

            for i in range(cores_used):
                self.mean_weights_multi.append(0)

            #creating batches [input data]
            data_length = int(len(training_inputs.getArray()[0]))
            left = left_size
            for i in range(int(cores_used)):
                mat = []
                dod = int(left_size - left)
                for j in range(batch_size):
                    mat.append(all_inputs[i * batch_size + j + dod])
                if (left > 0):
                    mat.append(all_inputs[i * batch_size + batch_size + dod])
                    left = left - 1
                    new_batch = Matrix.Matrix("",batch_size + 1,data_length,mat)
                else:
                    new_batch = Matrix.Matrix("",batch_size,data_length,mat)
                batches_in.append(self.Matrix_to_cupy_single(new_batch))

            #creating batches [output data]
            data_length = int(len(training_outputs.getArray()[0]))
            left = left_size
            for i in range(int(cores_used)):
                mat = []
                dod = int(left_size - left)
                for j in range(batch_size):
                    mat.append(all_outputs[i * batch_size + j + dod])
                if (left > 0):
                    mat.append(all_outputs[i * batch_size + batch_size + dod])
                    left = left - 1
                    new_batch = Matrix.Matrix("",batch_size + 1,data_length,mat)
                else:
                    new_batch = Matrix.Matrix("",batch_size,data_length,mat)
                batches_out.append(self.Matrix_to_cupy_single(new_batch))

            #starting multithreaded server
            hidden_Layout_count = self.layers_count - 1

            
            mempool = cp.get_default_memory_pool() # GPU memory

            from msvcrt import getch, kbhit
            Iter = self.iter * 5
            modulo = 5 * ((iterations / Iter)/100)

            print("Iter: ",Iter)
            print("Press any key to end training")
            print("[ ",end="")
            freeze_support()
            loss_value = 0.0
            skutecznosc = 0.0

            lock = Lock()
            czas = time.perf_counter()

            #Fast learning variables
            if (cores_used > 10):
                Batch_Size = 10
            else:
                Batch_Size = cores_used
            ID = 0


            for j in range(int(iterations/Iter)):
                #wypisywanie postępu
                if (j%modulo == 0):
                    print(str(round((j*100)/(iterations/Iter),0))+"% ",end="",flush=True)
                    loss_value, skutecznosc = self.test_loss_eCuda(t_in,t_out,self.average_weight_cuda,sigmoid)
                    print("Loss: ",loss_value," Skutecznosc: ",skutecznosc,"%",end="",flush=True)
                    
                    if (j != 0):
                        durationTh = (time.perf_counter() - czas)
                        if (durationTh < 60):
                            print(" Time: ",str(durationTh),"s")
                        else:
                            minutes = int(durationTh / 60)
                            seconds = durationTh % 60
                            print(" Time: ",str(minutes),"min",str(seconds),"s")
                        czas = time.perf_counter()
                    else:
                        print("")
                    

                weights = []
                for i in range(hidden_Layout_count):
                    if (self.float_reduction):
                        weights.append(cp.zeros((self.neuron_inputs[i] * self.neuron_count[i]),dtype=cp.float64).reshape(self.neuron_count[i],self.neuron_inputs[i]))
                    else:
                        weights.append(cp.zeros((self.neuron_inputs[i] * self.neuron_count[i]),dtype=cp.float32).reshape(self.neuron_count[i],self.neuron_inputs[i]))

                #single thread
                
                for i in range(Batch_Size):
                    self.CUDA_train_batch(batches_in[ID],batches_out[ID],Iter,ID,sigmoid,sigmoid_derivative,sigmDot,lock)
                    ID += 1
                    if (ID >= cores_used):
                        ID = 0
                    if (kbhit()):
                        print(Fore.RED + " [przerwanie]" + Fore.RESET, end="",flush=True)
                        break
                if (kbhit()):
                    break

                #combine batches
                
                if (self.arythmetic_mean):  
                    #średnia arytmetyczna
                    for z in range(hidden_Layout_count):
                        iterator = (ID - Batch_Size) % cores_used
                        for i in range(Batch_Size):
                            weights[z] += self.all_synaptic_batches[iterator * hidden_Layout_count + z]
                            iterator = (iterator + 1)%cores_used
                        weights[z] = weights[z]/Batch_Size
                        self.average_weight_cuda[z] = weights[z]
                else:   
                    #średnia ważona
                    for z in range(hidden_Layout_count):
                        mean_weights = []
                        mean_weights_sum = 0.0

                        iterator = (ID - Batch_Size) % cores_used
                        for i in range(Batch_Size):
                            mean_weights.append(1 / self.create_loss_weights_cuda(t_in,t_out,self.all_synaptic_batches,hidden_Layout_count,iterator,sigmoid))
                            weights[z] += (self.all_synaptic_batches[iterator * hidden_Layout_count + z] * mean_weights[i])
                            mean_weights_sum += mean_weights[i]
                            iterator = (iterator + 1) % cores_used

                        weights[z] = weights[z]/mean_weights_sum

                        self.average_weight_cuda[z] = weights[z]

                

            print(" 100% ]")
            self.all_layer_weights = self.Convert_Weights(self.average_weight_cuda)
            self.synaptic_weights = self.Convert_Weights(self.average_weight_cuda)
            
            
            print(Fore.GREEN + "Training is done" + Fore.RESET)
            mempool.free_all_blocks() #free memory
            ex = Export(self.name)
            ex.save_weights(self.all_layer_weights)

    def CUDA_train_Server(self,training_inputs:Matrix.Matrix,training_outputs:Matrix.Matrix,iterations:int):
        print(Fore.LIGHTBLUE_EX + "Starting CUDA server..." + Fore.RESET)
        cp.cuda.Device(0).use()

        #################CUDA KERNELS######################
        if (self.float_reduction):   
            sigmoid = cp.ElementwiseKernel(
                'float64 in',
                'float64 out',
                '''
                float h = exp(-1 * in);
                out = 1 / (1 + h);
                ''',
                'sigmoid'
            )
            sigmoid_derivative = cp.ElementwiseKernel(
                'float64 in',
                'float64 out',
                '''
                out = in * (1 - in);
                ''',
                'sigmoid_derivative'
            )
        else:
            sigmoid = cp.ElementwiseKernel(
                'float32 in',
                'float32 out',
                '''
                float h = exp(-1 * in);
                out = 1 / (1 + h);
                ''',
                'sigmoid'
            )
            sigmoid_derivative = cp.ElementwiseKernel(
                'float32 in',
                'float32 out',
                '''
                out = in * (1 - in);
                ''',
                'sigmoid_derivative'
            )
        ###################################################

        #Przygotowanie zmiennych
        batches_in = []
        batches_out = []
        data_count = int(len(training_inputs.getArray()))
        cpu_count = self.automatic_thread_count_gpu(data_count)
        all_inputs, all_outputs, all_synaptic_weights, t_in, t_out = self.Matrix_to_cupy(training_inputs,training_outputs,self.all_layer_weights)
        self.all_synaptic_weights = (all_synaptic_weights)


        #przechodzenie na tryb jednordzeniowy gdy mało danych
        if (cpu_count == 1):
            self.CUDA_train(training_inputs,training_outputs,iterations)
            print(Fore.GREEN + "Training is done" + Fore.RESET)
            ex = Export(self.name)
            ex.save_weights(self.all_layer_weights)
        else:
            #ustalenie ilości używanych wątków
            if (data_count >= cpu_count):
                batch_size = int(data_count / cpu_count)
                left_size = data_count - cpu_count*batch_size
                cores_used = cpu_count
            else:
                batch_size = 1
                left_size = 0
                cores_used = data_count

            print(Fore.LIGHTBLUE_EX + "\n[Creating Batches]" + Fore.RESET)
            #print("CPU Threads: ",os.cpu_count())
            print("Data count: ",data_count)
            print("Batch size: ",batch_size)
            print("Left size: ",left_size)
            print("Threads used: ",cores_used)
            print("")

            #pamięć współdzielona
            manager = Manager()
            self.synaptic_batches = manager.list()
            self.all_synaptic_batches = manager.list()

            for i in range(self.layers_count - 1):
                self.synaptic_batches.append(self.all_synaptic_weights[i])
            
            self.average_weight_cuda = manager.list()
            
            for i in range(self.layers_count - 1):
                self.average_weight_cuda.append(self.all_synaptic_weights[i])


            for i in range(cores_used * (self.layers_count - 1)):
                self.all_synaptic_batches.append(self.average_weight_cuda[i % (self.layers_count - 1)])

            for i in range(cores_used):
                self.mean_weights_multi.append(0)

            #creating batches [input data]
            data_length = int(len(training_inputs.getArray()[0]))
            left = left_size
            for i in range(int(cores_used)):
                mat = []
                dod = int(left_size - left)
                for j in range(batch_size):
                    mat.append(all_inputs[i * batch_size + j + dod])
                if (left > 0):
                    mat.append(all_inputs[i * batch_size + batch_size + dod])
                    left = left - 1
                    new_batch = Matrix.Matrix("",batch_size + 1,data_length,mat)
                else:
                    new_batch = Matrix.Matrix("",batch_size,data_length,mat)
                batches_in.append(self.Matrix_to_cupy_single(new_batch))

            #creating batches [output data]
            data_length = int(len(training_outputs.getArray()[0]))
            left = left_size
            for i in range(int(cores_used)):
                mat = []
                dod = int(left_size - left)
                for j in range(batch_size):
                    mat.append(all_outputs[i * batch_size + j + dod])
                if (left > 0):
                    mat.append(all_outputs[i * batch_size + batch_size + dod])
                    left = left - 1
                    new_batch = Matrix.Matrix("",batch_size + 1,data_length,mat)
                else:
                    new_batch = Matrix.Matrix("",batch_size,data_length,mat)
                batches_out.append(self.Matrix_to_cupy_single(new_batch))

            #starting multithreaded server
            hidden_Layout_count = self.layers_count - 1

            
            mempool = cp.get_default_memory_pool() # GPU memory

            from msvcrt import getch, kbhit
            Iter = self.iter * 5
            modulo = 5 * ((iterations / Iter)/100)

            print("Iter: ",Iter)
            print("Press any key to end training")
            print("[ ",end="")
            freeze_support()
            loss_value = 0.0
            skutecznosc = 0.0

            lock = Lock()
            czas = time.perf_counter()
            for j in range(int(iterations/Iter)):
                #wypisywanie postępu
                if (j%modulo == 0):
                    print(str(round((j*100)/(iterations/Iter),0))+"% ",end="",flush=True)
                    loss_value, skutecznosc = self.test_loss_eCuda(t_in,t_out,self.average_weight_cuda,sigmoid)
                    print("Loss: ",loss_value," Skutecznosc: ",skutecznosc,"%",end="",flush=True)
                    
                    if (j != 0):
                        print(" Time: ",(time.perf_counter() - czas),"s")
                        czas = time.perf_counter()
                    else:
                        print("")
                    

                weights = []
                for i in range(hidden_Layout_count):
                    if (self.float_reduction):
                        weights.append(cp.zeros((self.neuron_inputs[i] * self.neuron_count[i]),dtype=cp.float64).reshape(self.neuron_count[i],self.neuron_inputs[i]))
                    else:
                        weights.append(cp.zeros((self.neuron_inputs[i] * self.neuron_count[i]),dtype=cp.float32).reshape(self.neuron_count[i],self.neuron_inputs[i]))

                #single thread
                
                for i in range(cores_used):
                    self.CUDA_train_batch(batches_in[i],batches_out[i],Iter,i,sigmoid,sigmoid_derivative,lock)
                    if (kbhit()):
                        print(Fore.RED + " [przerwanie]" + Fore.RESET, end="",flush=True)
                        break

                if(kbhit()):
                    break


                #combine batches
                
                #średnia ważona
                for z in range(hidden_Layout_count):
                    mean_weights = []
                    mean_weights_sum = 0.0

                    for i in range(cores_used):
                        mean_weights.append(1 / self.create_loss_weights_cuda(t_in,t_out,self.all_synaptic_batches,hidden_Layout_count,i,sigmoid))
                        weights[z] += (self.all_synaptic_batches[i * hidden_Layout_count + z] * mean_weights[i])
                        mean_weights_sum += mean_weights[i]

                    weights[z] = weights[z]/mean_weights_sum

                    self.average_weight_cuda[z] = weights[z]

                

            print(" 100% ]")
            self.all_layer_weights = self.Convert_Weights(self.average_weight_cuda)
            self.synaptic_weights = self.Convert_Weights(self.average_weight_cuda)
            
            print(Fore.GREEN + "Training is done" + Fore.RESET)
            mempool.free_all_blocks() #free memory
            ex = Export(self.name)
            ex.save_weights(self.all_layer_weights)



    def create_loss_weights_cuda(self,test_inputs,test_outputs,synaptic_weights,layers,ID,sigmoid):
        data_count = len(test_inputs)
        wyniki = []

        for i in range(data_count):
            wyniki.append(self.think_CUDA(test_inputs[i],synaptic_weights,sigmoid,layers,ID))

        suma = 0.0
        for i in range(data_count):
            suma += (cp.square(test_outputs[i] - wyniki[i])).mean(axis=None)

        mianownik = float(data_count)
        
        self.mean_weights_multi[ID] = 1 / (suma / mianownik)
        return suma/mianownik


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

    def automatic_thread_count_gpu(self,data_count:int):
        if (data_count > 90):
            output = 1
        else:
            output = 0
        mnoznik = 62

        if (not(self.force == True and self.threads > 0)):   
            for i in range(data_count):
                if (i % int(mnoznik) == 0):
                    output += 1
        else:
            output = self.threads

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
            print(Fore.GREEN + "Training is done" + Fore.RESET)

            ex = Export(self.name)
            ex.save_weights(self.all_layer_weights)
        else:
            if (data_count >= cpu_count):
                batch_size = int(data_count / cpu_count)
                left_size = data_count - cpu_count*batch_size
                cores_used = cpu_count
            else:
                batch_size = 1
                left_size = 0
                cores_used = data_count

            print(Fore.LIGHTBLUE_EX + "\n[Creating Batches]" + Fore.RESET)
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

            
            hidden_Layout_count = self.layers_count - 1
            string = []
            for i in range(hidden_Layout_count):
                string.append(Matrix.Matrix("",self.neuron_inputs[i],self.neuron_count[i],None).T().getString())
            
            modulo = 5 * ((iterations / 100)/100)

            from msvcrt import getch, kbhit

            Iter = self.iter #domyślnie 100
            #new multithreaded Loop
            print("Iter: ",Iter)
            print("Press any key to end training")
            print("[ ",end="")
            freeze_support()
            loss_value = 0.0
            skutecznosc = 0.0
            for j in range(int(iterations/Iter)):
                #wypisanie postępu
                if (j%modulo == 0):
                    print(str(round((j*100)/(iterations/Iter),0))+"% ",end="",flush=True)
                    loss_value, skutecznosc = self.test_loss_extended(self.test_inputs,self.test_outputs)
                    print("Loss: ",loss_value," Skutecznosc: ",skutecznosc,"%",flush=True)

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
                    print(Fore.RED + " [przerwanie] " + Fore.RESET, end="", flush=True)
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

            print(Fore.GREEN + "Training is done" + Fore.RESET)

            ex = Export(self.name)

            ex.save_weights(self.all_layer_weights)

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
            