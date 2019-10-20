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

    #load network from file
    net.load_synaptic_weights(Matrix.Matrix("[1,2][-10,35][22,56]"))
    net.print_synaptic_weights()

    import os
    os.system("pause")

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