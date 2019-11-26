from Matrix import Matrix
import NeuralNetwork
import NeuNet
import Import

from colorama import init, Fore, Back, Style
   
def hidden_Layout(data:str):
    output = Matrix(data)
    return output

def Beep(duration = 0.40):
    import winsound
    frequency = 800
    Duration = int(1000 * duration)
    winsound.Beep(frequency, Duration)

#main
if __name__ == '__main__':
    init() #For colors

    experimental = True

    if (1):
        net = NeuNet.NeuNet() #dla IRIS
    else:
        net = NeuNet.NeuNet(True,False) #dla kr-vs-kp, kółko i krzyżyk

    '''
    name = "GamesForAi"
    net_name = "GamesForAi-gpu"
    '''
    '''
    name = "abalone"
    net_name = "Abalone-gpu-2"   #if (1)
    '''
    '''
    #if (1)  #and i or tak samo
    name = "xor"
    net_name = "Xor"
    '''
    '''
    #if (0) 60000 iteracji
    name = "kr-vs-kp"
    net_name = "Kr_test_1"
    '''
    name = "breast_cancer"
    net_name = "Breast_Cancer"

    im = Import.Import("INPUT\Training_Data\\" + name + ".txt") #import training data

    net.input(im.get_input())
    net.output(im.get_output())
    net.labels(im.get_labels())

    net.iterations(60000) #30000 iris
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
    #net.add_hidden_layout(Matrix("[0][7][5][9][7][7][5][5][6][0]")) #52%
    #net.add_hidden_layout(Matrix("[0][7][5][9][7][7][5][5][6][12][0]"))
    #net.add_hidden_layout(Matrix("[0][7][5][9][0]")) #65% dla 48thr 25000/2   i arytmetyczna średnia
    #net.add_hidden_layout(Matrix("[0][25][12][12][12][5][5][5][5][0]")) # O i X
    #net.add_hidden_layout(Matrix("[0][4][0]")) #Xor,And,Or 100% 10000 iteracji

    #breast cancer
    #net.add_hidden_layout(Matrix("[0][64][64][32][32][9][5][1][0]"))
    #net.add_hidden_layout(Matrix("[0][31][15][5][1][0]"))
    net.add_hidden_layout(Matrix("[0][32][16][1]")) #82% dla [10] i iter 20
    

    net.Setup()
    net.set_name(net_name) #set name for the network
    net.set_iter(100) #100
    net.set_device("gpu") # "cpu" "gpu"

    #Speed setting
    net.force_arythmetic_mean(True)
    net.force_float_reduction(True) #float32 vs float64




    #load network from file
    '''
    nim = Import.NetImport("OUTPUT\Saved_Networks\\" + net_name + ".txt")
    net.load_synaptic_weights(nim.get_weights())
    net.print_synaptic_weights()'''

    if (experimental):  
        net.Train()
        net.Think_from_File(tin.get_input_matrix(),tin.get_output_matrix(),net_name,im.get_labels_matrix())
        Beep()
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