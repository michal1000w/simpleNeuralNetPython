import Matrix
import NeuralNetwork
import NeuNet
import Import
   

#main
if __name__ == '__main__':  
    experimental = True

    net = NeuNet.NeuNet()

    im = Import.Import("INPUT\data3.txt")

    net.input(im.get_input())
    net.output(im.get_output())
    net.labels(im.get_labels())

    net.iterations(10000)
    #net.seed(4) #optional

    '''net.input("[1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4][1,2,3][1,2,3][2,3,4][4,4,4]")
    net.output("[1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0][0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]")
    
    net.input("[1,2,3][1,2,3][2,3,4][4,4,4]")
    net.output("[1,1,0,0][0,0,1,1]")
    
    net.labels("[japko][pomaranicz]")'''

    #experimental
    net.go_experimental(experimental)
    net.set_threads(0)

    net.Setup()

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
            print("Training Failed")