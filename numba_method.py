from numba import njit, prange

njit_parallel = njit(cache = True, parallel = True)


def updateHy(grid):
    return _updateHy(grid.ez, grid.hy, grid.chyh, grid.chye, grid.size)

@njit_parallel
def _updateHy(ez, hy, chyh, chye, SIZE):
    for mm in prange(0, SIZE - 1, 1):
        hy[mm] = chyh[mm] * hy[mm] + chye[mm] * (ez[mm + 1] - ez[mm])# / imp0 #/ muR[mm]
    return

def updateEz(grid):
    return _updateEz(grid.ez, grid.hy, grid.cezh, grid.ceze, grid.size)

@njit_parallel
def _updateEz(ez, hy, cezh, ceze, SIZE):
    for mm in prange(1, SIZE - 1, 1):
        ez[mm] = ceze[mm] * ez[mm] + cezh[mm] * (hy[mm] - hy[mm - 1])# * imp0 / epsR[mm] #ABC boundary removed
    return
