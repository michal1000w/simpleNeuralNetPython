import Matrix
import NeuralNetwork
import NeuNet

#main
if __name__ == '__main__':  
    net = NeuNet.NeuNet()

    #net.input("[1,2,3][1,2,3][2,3,4][4,4,5][6,6,7][5,3,2][3,2,1][1,1,-1] [1,2,3][1,2,3][2,3,4][4,4,5][6,6,7][5,3,2][3,2,1][1,1,-1]")
    #net.output("[1,1,0,1,0,1,1,1,0,0,1,0,0,1,1,1][0,0,1,1,0,0,1,1,0,1,0,0,1,1,1,1]")
    
    net.input("[1,2,3][1,2,3][2,3,4][4,4,4]")
    net.output("[1,1,0,0][0,0,1,1]")

    net.iterations(10000)
    #net.seed(4) #optional
    net.labels("[japko][pomaranicz]")

    #experimental
    #net.go_experimental(True)
    net.set_threads(1)

    net.Setup()


    '''net.Train()
    net.Think("[1,13,4]")
    net.Think("[5,3,2]")'''
    try:
        net.Train()

        net.Think("[1,13,4]")
        net.Think("[5,3,2]")
    except:
        print("Training Failed")