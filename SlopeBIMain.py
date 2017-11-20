#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:06:43 2017
Main Script Compiled to Keep Track of Different Functions

@author: jacob
"""

# PARAMETER SPACE EXPLORATION
QGPlots.py # Makes the Mechoso 1980 plots of growth rate vs. delta for the QG limit.

SIRegimeScan.py # Scan SI/CI parameter space for energetics.
SIRegimeScanPlot.py # Load output of above, make some plots (not used)
RegimeDiagram.py # Make the regime diagram

SlopeBIParallelStone.py # Calculates growth rates varying theta (delta) for fixed N2, Uz
SlopeSIParallel.py # Calculates growth rates for symmetric mode varying theta (delta)
StabilityContour.py # Plots for the above stability calculations.

SlopeBIParallelRi.py # Calculates growth rate at fixed theta, N2 and varying Ri
StabilityContourRI.py # Plots for above

DeepShallowCalc.py# Calculate growth rate for deep and shallow modes as a function of slope angle.
Stabilityplots.py # Plot the growth rates and Ri structure for the above.

SpecificWavenumberDelta.py # Plot Energetics and Perturbation structure at given wavenumber/delta

# EKMAN PROBLEM
EkmanGammaCalcBI.py # Varies gamma and calculates BI growth rates
EkmanGammaCalcSI.py # varies gamma and calculates SI growth rates
StabilityContourEkGamma.py # Makes plot of BI and SI modes.


# MIXING-DRIVEN
MixingStability.py # Calculates 1D mixing-problem, then checks stability characteristics.


## OLD STUFF NOT IN USE
EkmanSolver.py # Version of Ekman solver written using Dedalus, doesn't work for time varying kappa
nonlinear_solver.py # 3D NS Solver, doesn't work very well.
bci.py # 3D NS Solver from Joern. Works ok until blow-up...
EkmanSolverForward.py # Integrates the 1D BBL Ekman problem forward in time, saves figs and files
SlopeBIParallelEkman.py # Uses output of EkmanSolverForward to calculate and save growth rates (call mpiexec -n 20 python3 ...)
StabilityPlotsEkman.py # Loads the output of SlopeBIParallelEkman and makes some plots
AllenBLStructure.py # Basic code to calculate BL flow and N2 structure based on Allen and Newberger 1998
