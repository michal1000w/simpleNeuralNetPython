import Matrix
import NeuralNetwork
import NeuNet
import Import
   

#main
if __name__ == '__main__':  
    experimental = True

    net = NeuNet.NeuNet()

    im = Import.Import("INPUT\Training_Data\data3.txt")

    net.input(im.get_input())
    net.output(im.get_output())
    net.labels(im.get_labels())

    net.iterations(10000)
    #net.seed(4) #optional

    #experimental
    net.go_experimental(experimental)
    net.set_threads(0)
    #net.force_threads(True)  #to set up more threads than CPU cores

    net.Setup()
    net.set_name("my_network_2")

    #load network from file
    '''nim = Import.NetImport("OUTPUT\Saved_Networks\my_network_1.txt")
    net.load_synaptic_weights(nim.get_weights())
    net.print_synaptic_weights()'''


    if (experimental):  
        net.Train()
        net.Think("[1,13,4]")
        net.Think("[5,3,2]")
    else:   
        try:
            net.Train()

            net.Think("[1,13,4]")
            net.Think("[5,3,2]")
        except:
            print("Training Failed (Maybe too much input data -> try multithreaded workflow)")