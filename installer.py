import ctypes,os
from os import system

def isAdmin():
    try:
        is_admin = (os.getuid() == 0)
    except AttributeError:
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    return is_admin

def isInstalled():
    try:
        #import pycuda
        import cupy
        return True
    except:
        return False


if __name__ == '__main__':
    if (not(isAdmin())):
        print("Run this script as administrator")
        system("pause")
    else:   
        if (isInstalled()):
            print("Already installed")
            system("pause")
        else:
            #system("pip install pycuda")
            system("pip install cupy-cuda101")
            print("[Done]")
            system("pause")