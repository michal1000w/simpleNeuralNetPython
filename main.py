import os
import random

class Matrix():
    #zmienne ogólnoklasowe
    mat = []
    wiersze = 0
    kolumny = 0

    def __init__(self,data):
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


#main

m = Matrix("[1,6][2,4]")
#m.print()
m.printMatrix()