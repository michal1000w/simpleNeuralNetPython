import random
import time
import math


class Matrix:
    #podstawowe funkcje

    def __init__(self,data:str):
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

    def printMatrix(self):
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
                helper[i][j] = str(self.mat[j][i])

        #print(helper)

        data = ""
        for i in range(x):
            data += "["
            for j in range(y):
                data += helper[i][j]
                if (j < y-1):
                    data += ","
            data += "]"

        #print(data)
        return Matrix(data)

    def exp(self,inverted = False):
        E = 1.78107

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
                    helper[i][j] = str(math.exp(-self.mat[i][j]))
        else :
            for i in range(y):
                for j in range(x):
                    helper[i][j] = str(math.exp(self.mat[i][j]))

        #print(helper)

        data = ""
        for i in range(y):
            data += "["
            for j in range(x):
                data += helper[i][j]
                if (j < x-1):
                    data += ","
            data += "]"

        #print(data)
        return Matrix(data)
    
    def sigmoid(self):
        y = self.wiersze
        x = self.kolumny

        helper = self.exp(True).mat

        for i in range(y):
            for j in range(x):
                helper[i][j] = 1 / (1 + helper[i][j])

        data = ""
        for i in range(y):
            data += "["
            for j in range(x):
                data += str(helper[i][j])
                if (j < x-1):
                    data += ","
            data += "]"

        return Matrix(data)

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

        data = ""
        for i in range(y):
            data += "["
            for j in range(x):
                data += str(helper[i][j])
                if (j < x-1):
                    data += ","
            data += "]"


        return Matrix(data)

    #przeciążenia operatorów
    def __add__(self, o):
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

            data = ""
            for i in range(y):
                data += "["
                for j in range(x):
                    data += str(helper[i][j])
                    if (j < x-1):
                        data += ","
                data += "]"
        except:
            data = "[]"
            print("Błąd przy dodawaniu")

        return Matrix(data)

    def __iadd__(self, o): # +=
        try:
            for i in range(self.wiersze):
                for j in range(self.kolumny):
                    self.mat[i][j] += o.mat[i][j]
        except:
            print("Błąd przy dodawaniu [+=]")
        return self

    def __sub__(self, o):
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

            data = ""
            for i in range(y):
                data += "["
                for j in range(x):
                    data += str(helper[i][j])
                    if (j < x-1):
                        data += ","
                data += "]"
        except:
            data = "[]"
            print("Błąd przy odejmowaniu")

        return Matrix(data)

    def __mul__(self, o): # * mnożenie macierzy
        try:
            y1 = self.wiersze
            x1 = self.kolumny
            y2 = o.wiersze
            x2 = o.kolumny

            helper = []

            z = 0   
            for i in range(y1):
                rowList = []
                for j in range(x2):
                    rowList.append(" ")
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

            data = ""
            for i in range(y1):
                data += "["
                for j in range(x2):
                    data += str(helper[i][j])
                    if (j < x2-1):
                        data += ","
                data += "]"

        except:
            data = "[]"
            print("Błąd przy mnożeniu macierzy")

        return Matrix(data)

    def __imul_(self, o): # *= mnożenie self.x1 * o.x1 ...
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

            data = ""
            for i in range(y):
                data += "["
                for j in range(x):
                    data += str(helper[i][j])
                    if (j < x-1):
                        data += ","
                data += "]"
        except:
            data = "[]"
            print("Błąd przy mnożeniu elementów [*=]")

        return Matrix(data)


#main

m = Matrix("[1,6,5][2,4,3]")
#m.print()
m.printMatrix()

#m.T().printMatrix()
#m.exp().printMatrix()
#m.exp(True).printMatrix()

#m.sigmoid().printMatrix()
#m.sigmoid_derivative().printMatrix()

'''c = Matrix("[2,3,6,7][4,5,8,9]")
(c-m).printMatrix()

(m + m).printMatrix()
(m - m).printMatrix()'''
(m*m).printMatrix()
(m*m.T()).printMatrix()