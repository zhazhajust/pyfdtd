#from functools import cache
import h5py
import numpy as np
#import matplotlib.pyplot as plt
from .numba_method import updateEz, updateHy
from .boundary import BoundaryComm, abc, abcInit, tfsfUpdate

SIZE = 350
maxTime = 1350
#LOSS = 0.02
#LOSS_LAYER = 360
imp0 = 377.
tfsf_location = 19

def writeH5(filename, timestep, data):
    try:
        with h5py.File(filename, "a") as f:
            f[f"Ez/{str(timestep).zfill(5)}"] = data
    except:
        with h5py.File(filename, "w") as f:
            f[f"Ez/{str(timestep).zfill(5)}"] = data
    return

def readH5(filename):
    data = []
    with h5py.File(filename, "r") as f:
        for timestep in f["Ez"].keys():
            #print(timestep)
            data.append(f[f"Ez/{timestep}"][:])
    return np.asarray(data)

'''
def InitCoef(imp0, sigma, cezh, ceze, chyh, chye):
    for mm in range(SIZE):
        if mm < 0:
            ceze[mm] = 1.0
            cezh[mm] = imp0
        elif mm < LOSS_LAYER:
            ceze[mm] = 1.0
            cezh[mm] = imp0 / sigma[mm] #9.0
        else:
            ceze[mm] = (1.0 - LOSS) / (1.0 + LOSS)
            cezh[mm] = imp0 / sigma[mm] / (1.0 + LOSS)
    
    for mm in range(SIZE - 1):
        if mm < 0:
            chyh[mm] = 1.0
            chye[mm] = 1.0 / imp0
        if mm < LOSS_LAYER:
            chyh[mm] = 1.0
            chye[mm] = 1.0 / imp0 / 1.0 
        else:
            chyh[mm] = (1.0 - LOSS) / (1.0 + LOSS)
            chye[mm] = 1.0 / imp0 / 1.0 / (1.0 + LOSS)
    return
'''

def InitCoef(imp0, sigma, grid):
    cezh, ceze, chyh, chye = grid.cezh, grid.ceze, grid.chyh, grid.chye
    for mm in range(SIZE):
            ceze[mm] = 1.0
            cezh[mm] = imp0 / sigma[mm]
    
    for mm in range(SIZE - 1):
            chyh[mm] = 1.0
            chye[mm] = 1.0 / imp0
    return

class Grid(object):
    def __init__(self, SIZE):
        self.size = SIZE
        
        self.ez = np.zeros(SIZE)
        self.hy = np.zeros(SIZE - 1)
        
        self.ceze = np.zeros(SIZE)
        self.cezh = np.zeros(SIZE)
        
        self.chye = np.zeros(SIZE - 1)
        self.chyh = np.zeros(SIZE - 1)
        return
    
def sigmaFunc(time):
    return np.linspace(1.0, 3.0, SIZE) + (time/150)

def main():
    #ez = np.zeros(SIZE)
    #hy = np.zeros(SIZE - 1)
    
    #ceze = np.zeros(SIZE)
    #cezh = np.zeros(SIZE)
    
    #chye = np.zeros(SIZE - 1)
    #chyh = np.zeros(SIZE - 1)
    grid = Grid(SIZE)
    comm = BoundaryComm(tfsf_location)
    
    for qTime in range(maxTime):
        #sigma = np.linspace(1.0, 3.0, SIZE) + (qTime/150) #+ np.exp(qTime/200) #**2
        sigma = sigmaFunc(qTime)
        #sigma = np.power(sigma, 2.0)
        InitCoef(imp0, sigma, grid)
        comm.abcInit(grid)
        #rhs ABC removed
        updateHy(grid)        
        comm.tfsfUpdate(grid, qTime)
        updateEz(grid)
        comm.abc(grid)
        writeH5("Ez.hdf5", qTime, grid.ez)
    return

if __name__ == "__main__":
    main()