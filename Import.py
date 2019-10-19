import Matrix

class Import:
    def __init__(self,path:str):
        self.Input = []
        self.Output = []
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

            i1 = 0
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