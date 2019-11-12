from Matrix import Matrix
import NeuralNetwork
import NeuNet
import Import
   
def hidden_Layout(data:str):
    output = Matrix(data)
    return output

#main
if __name__ == '__main__':  
    experimental = True

    if (0):
        net = NeuNet.NeuNet() #dla IRIS
    else:
        net = NeuNet.NeuNet(True,False) #dla kr-vs-kp, kółko i krzyżyk

    name = "GamesForAi"
    net_name = "GamesForAi-gpu"


    im = Import.Import("INPUT\Training_Data\\" + name + ".txt") #import training data

    net.input(im.get_input())
    net.output(im.get_output())
    net.labels(im.get_labels())

    net.iterations(10000) #30000 iris
    #net.seed(4) #optional

    #experimental
    net.go_experimental(experimental)
    net.print_synaptic_set(False)

    net.set_threads(0) #196 dla GamesForAI
    net.force_threads(True)  #to set up more threads than CPU cores

    tin = Import.Import("INPUT\Test_Data\\" + name + ".txt") #import test_data
    net.add_testing_data(tin.get_input_matrix(),tin.get_output_matrix())

    
    #net.add_hidden_layout(Matrix("[0][6][0]"))
    #net.add_hidden_layout(Matrix("[0][5][0]")) #dla kr-vs-kp
    #net.add_hidden_layout(Matrix("[0][6][2][0]"))
    #net.add_hidden_layout(Matrix("[0][0]")) #no hidden
    #net.add_hidden_layout(Matrix("[0][5][3][2][3][3][0]")) #niby abalone ale nie działa za dobrze
    #net.add_hidden_layout(Matrix("[0][5][3][2][0]"))
    net.add_hidden_layout(Matrix("[0][25][12][12][12][5][5][5][5][0]"))
    

    net.Setup()
    net.set_name(net_name) #set name for the network
    net.set_iter(100) #100
    net.set_device("gpu") # "cpu" "gpu"

    #load network from file
    '''nim = Import.NetImport("OUTPUT\Saved_Networks\\" + net_name + ".txt")
    net.load_synaptic_weights(nim.get_weights())
    net.print_synaptic_weights()'''

    if (experimental):  
        net.Train()
        net.Think_from_File(tin.get_input_matrix(),tin.get_output_matrix(),net_name,im.get_labels_matrix())
        
    else:   
        net.Train()
        net.Think_from_File(tin.get_input_matrix(),tin.get_output_matrix(),net_name,im.get_labels_matrix())
        '''try:
            net.Train()

            net.Think_from_File(tin.get_input_matrix(),tin.get_output_matrix(),name,im.get_labels_matrix())
        except Exception as e:
            print("Training Failed (Maybe too much input data -> try multithreaded workflow)")
            print(e)
        '''