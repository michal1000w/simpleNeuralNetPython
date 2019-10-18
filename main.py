import os
import random

class Matrix():
    #zmienne ogólnoklasowe
    mat = []
    wiersze = 0
    kolumny = 0

    def __init__(self,data,rowCount,colCount):
        self.wiersze = rowCount
        self.kolumny = colCount
        for i in range(rowCount):
            rowList = []
            for j in range(colCount):
                rowList.append(data[colCount * i + j])
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
m = Matrix([1,2,3,4,5,6],3,2)
m.print()
m.printMatrix()