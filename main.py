import Matrix
import NeuralNetwork
import NeuNet
import Import
   

#main
if __name__ == '__main__':  
    experimental = True

    net = NeuNet.NeuNet()

    im = Import.Import("INPUT\Training_Data\iris.txt")

    net.input(im.get_input())
    net.output(im.get_output())
    net.labels(im.get_labels())

    net.iterations(30000)
    #net.seed(4) #optional

    #experimental
    net.go_experimental(experimental)
    net.set_threads(2)
    #net.force_threads(True)  #to set up more threads than CPU cores

    net.Setup()
    net.set_name("Iris")

    #load network from file
    '''nim = Import.NetImport("OUTPUT\Saved_Networks\my_network_1.txt")
    net.load_synaptic_weights(nim.get_weights())
    net.print_synaptic_weights()'''


    if (experimental):  
        net.Train()
        net.Think("[5.5,3.6,1.4,0.1]")
        net.Think("[6.7,2.9,4.4,1.2]")
    else:   
        try:
            net.Train()

            net.Think("[5.5,3.6,1.4,0.1]")
            net.Think("[6.7,2.9,4.4,1.2]")
        except:
            print("Training Failed (Maybe too much input data -> try multithreaded workflow)")