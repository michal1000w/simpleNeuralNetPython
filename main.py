import Matrix
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
    net.set_threads(8)
    #net.force_threads(True)  #to set up more threads than CPU cores

    net.Setup()
    net.set_name("Kr-vs-Kp")

    #load network from file
    '''nim = Import.NetImport("OUTPUT\Saved_Networks\my_network_1.txt")
    net.load_synaptic_weights(nim.get_weights())
    net.print_synaptic_weights()'''


    if (experimental):  
        net.Train()
        '''net.Think("[5.5,3.6,1.4,0.1]")
        net.Think("[6.7,2.9,4.4,1.2]")'''
        net.Think("[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0]") #win
        net.Think("[0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0]") #lose
    else:   
        try:
            net.Train()

            net.Think("[5.5,3.6,1.4,0.1]")
            net.Think("[6.7,2.9,4.4,1.2]")
        except:
            print("Training Failed (Maybe too much input data -> try multithreaded workflow)")