import numpy as np

delay = 30.0
width = 40.0

def ezInc(time, location):
    return np.sin( -(time - location - delay)/(width/2) * 2 * np.pi) * np.exp(-(time - location - delay)**2/(width))
