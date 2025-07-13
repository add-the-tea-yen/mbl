import numpy as np
from genmulti import EDUM
from plot import plotAreaVolume

#plotAreaVolume('./spins/half/L14/psi.csv')
L = [x for x in range(7,16)]
for l in L:
    plotAreaVolume(f'./spins/half/L{l}_psi.csv')

#plotEchoTime('./spins/half/L14/phases.csv','./spins/half/L14/psi.csv',14)
