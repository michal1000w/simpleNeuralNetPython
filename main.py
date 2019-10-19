import Matrix
import NeuralNetwork
import NeuNet


#main

net = NeuNet.NeuNet()

net.input("[1,2,3][1,2,3][2,3,4]")
net.output("[1,1,0][0,0,1]")
net.iterations(1000)
net.labels("[japko][pomaranicz]")
net.Setup()
net.Train()

net.Think("[1,13,4]")