from .laser import ezInc
import numpy as np
from numba import njit

class BoundaryComm(object):
    def __init__(self, tfsfBoundary = 19):
        self.tfsfBoundary = tfsfBoundary
        self.ezOldLeft1 = np.zeros(3)
        self.ezOldLeft2 = np.zeros(3)

        self.ezOldRight1 = np.zeros(3)
        self.ezOldRight2 = np.zeros(3)

        self.abcCoefLeft = np.zeros(3)
        self.abcCoefRight = np.zeros(3)
        return
    
    def abcInit(self, grid):
        cezh, chye, size = grid.cezh, grid.chye, grid.size
        """
        Initial the 2 order abc boundary.

        Parameter
        ---------
        fieldtype :
            cezh: nd_array
            chye: nd_array
            size: int
        """
        abcInit(cezh, chye, size, self.abcCoefLeft, self.abcCoefRight)
        return
    
    def abc(self, grid):
        ez, size = grid.ez, grid.size
        """
        Impact 2 order abc boundary. Need 2 passed timestep data.

        Parameter
        ---------
        fieldtype :
            ez: nd_array
            size: int
        """
        abc(ez, size, self.ezOldLeft1, self.ezOldLeft2, self.ezOldRight1,
        self.ezOldRight2, self.abcCoefLeft, self.abcCoefRight)
        return
    
    def tfsfUpdate(self, grid, qTime):
        hy, ez, chye = grid.hy, grid.ez, grid.chye
        tfsfUpdate(hy, ez, chye, qTime, self.tfsfBoundary)
        return
    
    def move_shift(self, grid, mov_win):
        time, dx, v, last_move_time, dt = \
            grid.time, grid.dx, mov_win.v, mov_win.last_move_time
        dt = time - last_move_time
        if v * dt < dx:
            return
        else:
            self.ezOldLeft1[1:] = self.ezOldLeft1[:-1]
            self.ezOldLeft2[1:] = self.ezOldLeft2[:-1]
            
            self.ezOldRight1[1:] = self.ezOldRight1[:-1]
            self.ezOldRight2[1:] = self.ezOldRight2[:-1]
            
            cezh, chye, size = grid.cezh, grid.chye, grid.size
            self.abcInit(cezh, chye, size)
            return

@njit(cache = True)
def abcInit(cezh, chye, size, abcCoefLeft, abcCoefRight):
    #initDone = 1
    #global abcCoefLeft, abcCoefRight
    #/* calculate coefficients on left end of grid */
    temp1 = np.sqrt(cezh[0] * chye[0]);
    temp2 = 1.0 / temp1 + 2.0 + temp1;
    abcCoefLeft[0] = -(1.0 / temp1 - 2.0 + temp1) / temp2;
    abcCoefLeft[1] = -2.0 * (temp1 - 1.0 / temp1) / temp2;
    abcCoefLeft[2] = 4.0 * (temp1 + 1.0 / temp1) / temp2;

    #/* calculate coefficients on right end of grid */
    temp1 = np.sqrt(cezh[size - 1] * chye[size - 2]);
    temp2 = 1.0 / temp1 + 2.0 + temp1;
    abcCoefRight[0] = -(1.0 / temp1 - 2.0 + temp1) / temp2;
    abcCoefRight[1] = -2.0 * (temp1 - 1.0 / temp1) / temp2;
    abcCoefRight[2] = 4.0 * (temp1 + 1.0 / temp1) / temp2;
    return

@njit(cache = True)
def abc(ez, size, ezOldLeft1, ezOldLeft2, ezOldRight1,
        ezOldRight2, abcCoefLeft, abcCoefRight):
    #ez[0] = ez[1]
    #global ezOldLeft1, ezOldLeft2, ezOldRight1, \
    #    ezOldRight2, abcCoefLeft, abcCoefRight
        
    #ez[0] = ezOldLeft + abcCoefLeft * (ez[1] - ez[0])
    #ezOldLeft = ez[1]
    #ez[size - 1] = ezOldRight + abcCoefRight * (ez[size - 2] - ez[size - 1])
    #ezOldRight = ez[size - 2]
    #print(abcCoefLeft)
    ez[0] = abcCoefLeft[0] * (ez[2] + ezOldLeft2[0]) \
    + abcCoefLeft[1] * (ezOldLeft1[0] + ezOldLeft1[2] - \
    ez[1] - ezOldLeft2[1]) \
    + abcCoefLeft[2] * ezOldLeft1[1] - ezOldLeft2[2]

    #/* ABC for right side of grid */
    ez[size - 1] =  \
    abcCoefRight[0] * (ez[size - 3] + ezOldRight2[0]) \
    + abcCoefRight[1] * (ezOldRight1[0] + ezOldRight1[2] - \
        ez[size - 2] - ezOldRight2[1]) \
            + abcCoefRight[2] * ezOldRight1[1] - ezOldRight2[2]

    #/* update stored fields */
    for mm in range(3):        #/*@ \label{abcsecondG} @*/
        ezOldLeft2[mm] = ezOldLeft1[mm]
        ezOldLeft1[mm] = ez[mm]

        ezOldRight2[mm] = ezOldRight1[mm]
        ezOldRight1[mm] = ez[size - 1 - mm]
    return

def tfsfUpdate(hy, ez, chye, qTime, tfsfBoundary):
    hy[tfsfBoundary] -= ezInc(qTime, 0.0) * chye[tfsfBoundary]
    ez[tfsfBoundary + 1] += ezInc(qTime + 0.5, -0.5)
    return
