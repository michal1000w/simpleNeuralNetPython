import math

class Matrix:
    #podstawowe funkcje

    def __init__(self,data:str,wiersze = 0,kolumny = 0,mat = []):
        self.mat = []
        if (wiersze != 0 and kolumny != 0 and mat != None):
            self.wiersze = wiersze
            self.kolumny = kolumny
            self.mat = mat
        elif (mat == None):
            self.wiersze = wiersze
            self.kolumny = kolumny
            z = 0
            for i in range(self.wiersze):
                rowList = []
                for j in range(self.kolumny):
                    rowList.append(0)
                    z += 1
                self.mat.append(rowList)
        else:   
            self.mat = []
            self.wiersze = 0
            self.kolumny = 0
            #"[1,2,1.4][3,9,3][41,2,55]"

            #zmienne początkowe
            length = len(data)
            fragment = ""
            fragmenty = []

            for i in range(length):
                if (data[i] == '['):
                    fragment = ""
                    while (i < length - 1):
                        i += 1
                        if (data[i] == ']'):
                            break
                        fragment += data[i]
                    fragmenty.append(fragment)

            self.wiersze = len(fragmenty)
            #print(fragmenty)
            
            #podział fragmentów na pojedyńcze elementy i sprawdzanie ilości kolumn
            wartosci = []
            elementy = 0

            for i in range(self.wiersze):
                length = len(fragmenty[i]) #długość stringa

                j1 = 0
                while(j1 < length):
                    fragment = ""
                    while (fragmenty[i][j1] != ',' and j1 < length):
                        fragment += fragmenty[i][j1]
                        j1 += 1
                        if (j1 == length):
                            break
                    if (fragment != ""):
                        wartosci.append(float(fragment))
                    j1 += 1

                if (i==0):
                    elementy = len(wartosci)

            self.kolumny = elementy

            #przenoszenie wartości do macierzy
            z = 0
            for i in range(self.wiersze):
                rowList = []
                for j in range(self.kolumny):
                    rowList.append(wartosci[z])
                    z += 1
                self.mat.append(rowList)

            wartosci.clear()
            fragmenty.clear()

    def print(self):
        print(self.mat)

    def printMatrix(self,miejsca = -1):
        if (miejsca < 0):   
            for i in range(self.wiersze):
                for j in range(self.kolumny):
                    if (j == 0 and self.kolumny != 1):
                        print("[",self.mat[i][j],",",end="",flush=True)
                    elif (j == 0):
                        print("[",self.mat[i][j],"]",end="",flush=True)
                    elif (j == self.kolumny-1):
                        print("",self.mat[i][j],"]",end="",flush=True)
                    else:
                        print("",self.mat[i][j],",",end="",flush=True)
                print("",flush=True)
        else:
            for i in range(self.wiersze):
                for j in range(self.kolumny):
                    if (j == 0 and self.kolumny != 1):
                        print("[",round(self.mat[i][j],miejsca),",",end="",flush=True)
                    elif (j == 0):
                        print("[",round(self.mat[i][j],miejsca),"]",end="",flush=True)
                    elif (j == self.kolumny-1):
                        print("",round(self.mat[i][j],miejsca),"]",end="",flush=True)
                    else:
                        print("",round(self.mat[i][j],miejsca),",",end="",flush=True)
                print("",flush=True)

    def add(self,data):
        length = len(data)
        fragment = ""
        fragmenty = []

        for i in range(length):
            if (data[i] == '['):
                fragment = ""
                while (i < length - 1):
                    i += 1
                    if (data[i] == ']'):
                        break
                    fragment += data[i]
                fragmenty.append(fragment)

        self.wiersze = len(fragmenty)
        #print(fragmenty)
        
        #podział fragmentów na pojedyńcze elementy i sprawdzanie ilości kolumn
        wartosci = []
        elementy = 0

        for i in range(self.wiersze):
            length = len(fragmenty[i]) #długość stringa

            j1 = 0
            while(j1 < length):
                fragment = ""
                while (fragmenty[i][j1] != ',' and j1 < length):
                    fragment += fragmenty[i][j1]
                    j1 += 1
                    if (j1 == length):
                        break
                if (fragment != ""):
                    wartosci.append(float(fragment))
                j1 += 1

            if (i==0):
                elementy = len(wartosci)

        self.kolumny = elementy

        #przenoszenie wartości do macierzy
        z = 0
        for i in range(self.wiersze):
            rowList = []
            for j in range(self.kolumny):
                rowList.append(wartosci[z])
                z += 1
            self.mat.append(rowList)

        wartosci.clear()
        fragmenty.clear()

    def getArray(self):
        return self.mat
    
    def getString(self):
        y = self.wiersze
        x = self.kolumny

        data = ""
        for i in range(y):
            data += "["
            for j in range(x):
                data += str(self.mat[i][j])
                if (j < x-1):
                    data += ","
            data += "]"

        return data
    #matma
    def T(self):
        y = self.wiersze
        x = self.kolumny

        helper = []

        z = 0
        for i in range(x):
            rowList = []
            for j in range(y):
                rowList.append(" ")
                z += 1
            helper.append(rowList)

        for j in range(y):
            for i in range(x):
                helper[i][j] = self.mat[j][i]

        #print(helper)
        return Matrix("",x,y,helper)

    def exp(self,inverted = False):
        y = self.wiersze
        x = self.kolumny

        helper = []

        z = 0
        for i in range(y):
            rowList = []
            for j in range(x):
                rowList.append(" ")
                z += 1
            helper.append(rowList)

        if (inverted):
            for i in range(y):
                for j in range(x):
                    helper[i][j] = math.exp(-self.mat[i][j])
        else :
            for i in range(y):
                for j in range(x):
                    helper[i][j] = math.exp(self.mat[i][j])

        #print(helper)
        return Matrix("",y,x,helper)
    
    def sigmoid(self):
        y = self.wiersze
        x = self.kolumny

        helper = self.exp(True).mat

        for i in range(y):
            for j in range(x):
                helper[i][j] = 1 / (1 + helper[i][j])

        return Matrix("",y,x,helper)

    def sigmoid_derivative(self):
        y = self.wiersze
        x = self.kolumny

        helper = []

        z = 0
        for i in range(y):
            rowList = []
            for j in range(x):
                rowList.append(self.mat[i][j] * (1 - self.mat[i][j]))
                z += 1
            helper.append(rowList)

        return Matrix("",y,x,helper)

    def mean(self): #średnia arytmetyczna
        output = 0.0
        count = self.wiersze + self.kolumny
        suma = 0.0

        for i in range(self.wiersze):
            for j in range(self.kolumny):
                suma += self.mat[i][j]

        output = float(suma / float(count))

        return output

    def square(self):
        helper = []

        y = self.wiersze
        x = self.kolumny

        for i in range(y):
            rowList = []
            for j in range(x):
                rowList.append(self.mat[i][j] ** 2.0)
            helper.append(rowList)

        return Matrix("",y,x,helper)

    #przeciążenia operatorów
    def __add__(self, o): # + dodawanie macierzy
        y = self.wiersze
        x = self.kolumny

        helper = []

        z = 0

        try:    
            for i in range(y):
                rowList = []
                for j in range(x):
                    rowList.append(self.mat[i][j] + o.mat[i][j])
                    z += 1
                helper.append(rowList)

            return Matrix("",y,x,helper)
        except:
            print("Błąd przy dodawaniu")
            return Matrix("[]")

    def __iadd__(self, o): # +=
        try:
            for i in range(self.wiersze):
                for j in range(self.kolumny):
                    self.mat[i][j] += o.mat[i][j]
        except:
            print("Błąd przy dodawaniu [+=]")
        return self

    def __sub__(self, o): # - odejmowanie macierzy
        y = self.wiersze
        x = self.kolumny

        helper = []

        z = 0
        try:    
            for i in range(y):
                rowList = []
                for j in range(x):
                    rowList.append(self.mat[i][j] - o.mat[i][j])
                    z += 1
                helper.append(rowList)

            return Matrix("",y,x,helper)
        except:
            print("Błąd przy odejmowaniu")
            return Matrix("[]")

    def __mul__(self, o): # * mnożenie macierzy
        y1 = self.wiersze
        x1 = self.kolumny
        y2 = o.wiersze
        x2 = o.kolumny

        helper = []

        z = 0   
        try:
            for i in range(y1):
                rowList = []
                for j in range(x2):
                    rowList.append(0)
                    z += 1
                helper.append(rowList)

            suma = 0.0
            w = k = 0
            t = o.kolumny * self.wiersze
            t1 = t

            while (t > 0):
                suma = 0.0
                for i in range(x1):
                    suma += self.mat[w][i] * o.mat[i][k]
                helper[w][k] = suma

                k = (k + 1) % x2
                t -= 1
                if (t%x2 == 0 and t != t1 and w < (y1 - 1)):
                    w += 1
                if (t == 0):
                    break

            return Matrix("",y1,x2,helper)

        except:
            print("Błąd przy mnożeniu macierzy")
            return Matrix("[]")

    def __pow__(self, o): # ** mnożenie self.x1 * o.x1 ...
        y = self.wiersze
        x = self.kolumny

        helper = []

        z = 0   
        try:
            for i in range(y):
                rowList = []
                for j in range(x):
                    rowList.append(self.mat[i][j] * o.mat[i][j])
                    z += 1
                helper.append(rowList)

            return Matrix("",y,x,helper)
        except:
            print("Błąd przy mnożeniu elementów [**]")
            return Matrix("[]")

    def __mod__(self, o): #  % mnożenie przez stałą
        y = self.wiersze
        x = self.kolumny

        helper = []

        z = 0

        try:    
            for i in range(y):
                rowList = []
                for j in range(x):
                    rowList.append(self.mat[i][j] * o)
                    z += 1
                helper.append(rowList)

            return Matrix("",y,x,helper)
        except:
            print("Błąd przy mnożeniu przez stałą [%]")
            return Matrix("[]")

    def __truediv__(self, o): # / dzielenie przez stałą
        y = self.wiersze
        x = self.kolumny

        helper = []

        z = 0

        try:    
            for i in range(y):
                rowList = []
                for j in range(x):
                    rowList.append(self.mat[i][j] / o)
                    z += 1
                helper.append(rowList)

            return Matrix("",y,x,helper)
        except:
            print("Błąd przy dzieleniu [/]")
            return Matrix("[]")
    

        y = self.wiersze
        x = self.kolumny

        helper = []

        import copy
        z = 0
        for i in range(y):
            rowList = []
            for j in range(x):
                rowList.append(copy.copy(self.mat[i][j]))
                z += 1
            helper.append(rowList)
        return Matrix("",y,x,helper)

