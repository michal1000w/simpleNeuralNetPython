import Matrix

class Import:
    def __init__(self,path:str):
        self.Input = []
        self.Output = []
        self.Labels = []
        self.filename = path
        self.data = ""

        try:
            #odczyt z pliku
            file = open(path,"r")
            data = file.read()
            file.close()

            #opracowanie danych
            length = len(data)
            fragment = ""
            fragmenty = []

            #getting labels
            i1 = 0
            while (i1 < length):    
                if (data[0] == '{'):
                    fragment = ""
                    while (i1 < length - 1):
                        i1 += 1
                        if (data[i1] == '}'):
                            break
                        fragment += data[i1]
                    self.Labels.append(fragment)
                if (len(self.Labels) != 0):
                    break
                i1 += 1
            
            #print(self.Labels)

            #getting matrixes
            fragmenty = []
            while (i1 < length):
                if (data[i1] == '['):
                    fragment = ""
                    while (i1 < length - 1):
                        i1 += 1
                        if (data[i1] == ']'):
                            break
                        fragment += data[i1]
                    fragmenty.append(fragment)
                i1 += 1

            #print(fragmenty)
            #Rozdzielanie fragmentów do odpowiednich list
            length = len(fragmenty)
            fragment = ""
            for i in range(length):
                if (i % 2 == 0):
                    fragment = "[" + fragmenty[i] + "]"
                    self.Input.append(fragment)
                else:
                    fragment = "[" + fragmenty[i] + "]"
                    self.Output.append(fragment)

            #print(self.Input)
            #print(self.Output)

        except:
            print("Can't open the file:",path)

    def get_input(self):
        length = len(self.Input)
        
        data = ""
        for i in range(length):
            data += self.Input[i]

        return data

    def get_output(self):
        length = len(self.Output)
        
        data = ""
        for i in range(length):
            data += self.Output[i]

        data = Matrix.Matrix(data).T().getString()

        return data

    def get_input_matrix(self):
        matrix = []
        mat = Matrix.Matrix(self.get_input())
        for i in range(len(self.Input)):
            arr = Matrix.Matrix("")
            arr.add(str(mat.getArray()[i]).replace(" ",""))
            matrix.append(arr)
            #matrix.append(Matrix.Matrix(self.get_input()))

        return matrix

    def get_output_matrix(self):
        matrix = []
        mat = Matrix.Matrix(self.get_output())
        mat = mat.T()
        for i in range(len(self.Output)):
            #matrix.append(Matrix.Matrix(self.get_output()))
            arr = Matrix.Matrix("")
            arr.add(str(mat.getArray()[i]).replace(" ",""))
            matrix.append(arr)
        return matrix

    def get_labels(self):
        length = len(self.Labels[0]) #długość stringa
        labels = []

        j1 = 0
        while(j1 < length):
            fragment = ""
            while (self.Labels[0][j1] != ',' and j1 < length):
                fragment += self.Labels[0][j1]
                j1 += 1
                if (j1 == length):
                    break
            if (fragment != ""):
                labels.append(fragment)
            j1 += 1

        data = ""
        for i in range(len(labels)):
            data += "[" + labels[i] + "]"

        #print(data)

        return data

class NetImport:
    def __init__(self,path:str):
        self.Weights = []
        self.filename = path
        self.data = ""

        try:
            #odczyt z pliku
            file = open(path,"r")
            data = file.read()
            file.close()

            #opracowanie danych
            length = len(data)
            fragment = ""

            i1 = 0
            while (i1 < length):
                if (data[i1] == '['):
                    fragment = ""
                    while (i1 < length - 1):
                        i1 += 1
                        if (data[i1] == ']'):
                            break
                        fragment += data[i1]
                    self.Weights.append(fragment)
                i1 += 1
        except:
            print("Can't open the file:",path)

    def get_weights(self):
        length = len(self.Weights)

        data = ""
        for i in range(length):
            data += "[" + self.Weights[i] + "]"
        
        output = Matrix.Matrix(data)
        return output

class Export:
    def __init__(self,filename:str):
        self.file_name = filename
        self.path = "OUTPUT\Saved_Networks\\" + filename + ".txt"

    def save_weights(self,weights:Matrix.Matrix):
        print("Saving weights...")
        print("Path: ",self.path)

        try:
            f = open(self.path,"w+")
            f.write(weights.getString())
            f.close()
            print("Saved")
        except:
            print("Writing failed")

class Think_File:
    def __init__(self,filename:str):
        self.filename = filename
        self.path = "OUTPUT\Labeled_Data\\" + filename + ".txt"
        self.info_path = "OUTPUT\Labeled_Data\\" + filename + "-info.txt"

    def save_think_output(self,inp:[],out:[],predicted:[]):
        print("Saving Think Output...")
        print("Path: ",self.path)

        data_len = int(len(inp))

        predicted_simple = self.simplify_predicted(predicted)
        try:
            f = open(self.path,"w+")
            f.write("[input][real output][predicted output]\n")
            for i in range(data_len):
                f.write(inp[i].getString() + " " + out[i].getString(True) + " " + predicted_simple[i].getString(True)+"\n")
            f.close()
            print("Saved")
        except Exception as e:
            print("Writing failed " + e)

        
        '''print("Saving Test Info...")
        print("Path: ", self.filename)

        try:
            f = open(self.info_path,"w+")
        except:
            print("Writing failed")'''

    def simplify_predicted(self,predicted:[]):
        print("Simplifying predicted model...")
        output_count = len(predicted[0].getArray()[0])
        predicted_count = len(predicted)

        output = []
        
        for i in range(predicted_count):
            max_val = max(predicted[i].getArray()[0])
            max_pos = 0

            for j in range(output_count):
                if (predicted[i].getArray()[0][j] == max_val):
                    max_pos = j
                    break
            
            #zapis wyniku
            data = "["
            for j in range(output_count):
                if (j == max_pos):
                    data += "1"
                else:
                    data += "0"
                if (j < output_count - 1):
                    data += ","
            data += "]"

            output.append(Matrix.Matrix(data))

        return output

