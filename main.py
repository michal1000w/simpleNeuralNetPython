from Matrix import Matrix
import NeuralNetwork
import NeuNet
import Import
   

#main
if __name__ == '__main__':  
    experimental = True

    #net = NeuNet.NeuNet() #dla IRIS
    net = NeuNet.NeuNet(True,False) #dla kr-vs-kp

    im = Import.Import("INPUT\Training_Data\kr-vs-kp.txt")

    net.input(im.get_input())
    net.output(im.get_output())
    net.labels(im.get_labels())

    net.iterations(30000) #30000 iris
    #net.seed(4) #optional

    #experimental
    net.go_experimental(experimental)
    net.set_threads(1)
    #net.force_threads(True)  #to set up more threads than CPU cores

    tin = Import.Import("INPUT\Test_Data\kr-vs-kp.txt") #import test_data
    net.add_testing_data(tin.get_input_matrix(),tin.get_output_matrix())

    net.Setup()
    net.set_name("Kr_test")

    #load network from file
    '''nim = Import.NetImport("OUTPUT\Saved_Networks\Abalone.txt")
    net.load_synaptic_weights(nim.get_weights())
    net.print_synaptic_weights()'''

    if (experimental):  
        net.Train()
        net.Think_from_File(tin.get_input_matrix(),tin.get_output_matrix())
        '''net.Think("[5.5,3.6,1.4,0.1]","[1,0,0]") #iris-setosa
        net.Think("[6.7,2.9,4.4,1.2]","[0,1,0]") #iris-versicolor
        '''
        net.Think("[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0]","[1,0]") #win
        net.Think("[0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0]","[0,1]") #lose
        
        '''net.Think("[0.44,0.365,0.125,0.516,0.2155,0.114,0.155,10]") #m
        net.Think("[0.605,0.5,0.185,1.1185,0.469,0.2585,0.335,9]") #f
        net.Think("[0.365,0.295,0.095,0.25,0.1075,0.0545,0.08,9]") #i
        '''
    else:   
        try:
            net.Train()

            net.Think_from_File(tin.get_input_matrix(),tin.get_output_matrix())
        except Exception as e:
            print("Training Failed (Maybe too much input data -> try multithreaded workflow)")
            print(e)