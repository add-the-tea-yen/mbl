# Many Body Localisation Simulations
Simple scripts to run and do statistics on many body localisation related Simulations. This repository contains the following:
- An implementation of the polynomailly filtered exact diagonalisation algorithm for floquet operators
- A simpler implementation of the floquet system with scipy
- Statistical scripts in plot.py 
- implementations of the Loschmidt Echo

The code for the floquet operators generate phases.csv and psi.csv files which can be used for statistical treatment with plot.py in main.py

## To do
- [ ] Extend POLFED functionality for more than one spin value 
- [ ] Mixed Spin Chains in POLFED
- [ ] Add automatic benchmarking for POLFED
- [ ] Abstractions 

## List of Functions and their usage
```py
from genmulti import EDUM
EDUM(L,s,phases,psi)
```
Description: provide L, the spin and the names of the phases and psi and U for that spin will be generated and dumped.
This is the aformentioned simpler implementation.

```py
from plot import plotPhases
from upolfed import run_level_spacings
#run
run_level_spacing(L=12, J=np.pi/4, b=np.pi/4, phi_tgt=np.pi/2, disorder=True, fphases="phases.csv",fpsi="psi.csv")
#plot
plotPhases('./phases.csv')
```
Description: This is an implementation of POLFED, ideally to be implemented with plot.py to check the distribution of eigenvalues on the complex circle. It required that you provide values to target where
the filtered targets for eigenphases (phi_tgt). Additionally k (filter sharpness) and nev(number of eigenvalues) are also parameters of the function. This does not work with mixed chains 
