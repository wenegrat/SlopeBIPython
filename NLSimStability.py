#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:13:37 2018

Calculate growth rate from given date in NL sim evolution.

@author: jacob
"""
import numpy as np
from mpi4py import MPI
CW = MPI.COMM_WORLD
import time
from pylab import *
from dedalus import public as de
from dedalus.extras import flow_tools
import scipy.integrate as integrate
import logging
logger = logging.getLogger(__name__)
import h5py

ly_global = np.logspace(-5, -3, 192)*2*np.pi

directoryname='/home/jacob/dedalus/NLSIM/'
filename = '/home/jacob/dedalus/NLSIM/snapshots/snapshots_s1.h5'
fil = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % fil.keys())
a_group_key = list(fil.keys())[0]

# Get the data
data = list(fil['tasks'])

Lz = 1000

f = 1e-4 # Coriolis parameter
#N2 = (12*f)**2
#N = (3.4e-3)
tht = 0.01 # slope angle
kap4 = 2e5
nu = 1e-5
kap = nu # Unit Pr

BLH = 100
VINT = 0.1
Shmag = VINT/BLH
Ri = 1.5
N = np.sqrt(Ri*Shmag**2+f*Shmag/np.sin(tht))

nz = 256

z_basis = de.Chebyshev('z', nz, interval=(0,Lz), dealias=1)

domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

z1 = domain.grid(0)


# DEFINE FIELDS
tn = 12 # Day 3 if 6 hour timesteps
V = domain.new_field(name='V')
Vz = domain.new_field(name='Vz')
Bz = domain.new_field(name='Bz')
B = domain.new_field(name='B')

V['g'] = fil['tasks']['V'][tn,0,0,:]  + np.mean(np.mean(fil['tasks']['v'][tn,:,:,:], axis=0), axis=0)
V.differentiate('z', out=Vz)

B['g'] = np.mean(np.mean(fil['tasks']['b'][tn,:,:,:], axis=0), axis=0)
B.differentiate('z', out=Bz)
Bz['g'] = fil['tasks']['Bz'][tn, 0, 0,:] + Bz['g']

problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p', 'uz', 'vz',
            'bz'], eigenvalue='omg', tolerance = 1e-10)
problem.parameters['tht'] = tht
problem.parameters['U'] = 0
problem.parameters['VI'] = V
problem.parameters['VIb'] = VINT
problem.parameters['Uz'] = 0
problem.parameters['Vz'] = Vz
problem.parameters['BZI'] = Bz # This is a perturbation (Bztotal = Bz + N**2)
problem.parameters['N'] = N
problem.parameters['f'] = f
problem.parameters['kap'] = kap
problem.parameters['nu'] = nu
problem.parameters['A4'] = kap4
problem.parameters['k'] = 0. # will be set in loop
problem.parameters['l'] = 0. # will be set in loop

# SUBSTITIONS
problem.substitutions['dx(A)'] = "1j*k*A"
problem.substitutions['dy(A)'] = "1j*l*A"
problem.substitutions['HV(A)'] = '-A4*(dx(dx(dx(dx(A)))) + 2*dx(dx(dy(dy(A)))) + dy(dy(dy(dy(A)))))' #Horizontal biharmonic diff
problem.substitutions['dt(A)'] = "-1j*omg*A"
problem.substitutions['D(A,Az)'] = 'kap*dz(Az) + dz(kap)*Az' # Vertical diffusion operator

# EQUATIONS
problem.add_equation('dt(u) - f*v*cos(tht) + dx(p) - b*sin(tht) - D(u,uz) - HV(u) + VIb*dy(u)  + VI*dy(u)=0')
problem.add_equation('dt(v) + f*u*cos(tht) + dy(p) - D(v,vz) - HV(v)  + VIb*dy(v) +  VI*dy(v) + w*dz(VI) =0')
problem.add_equation('dz(p) - b*cos(tht) = 0')
problem.add_equation('dt(b) + u*N**2*sin(tht) + w*N**2*cos(tht) - D(b,bz) - HV(b) +VIb*dy(b)+ VI*dy(b) + w*BZI =0')
problem.add_equation('dx(u) + dy(v) + dz(w) = 0')



problem.add_equation('uz - dz(u) = 0')
problem.add_equation('vz - dz(v) = 0')
#problem.add_equation('wz - dz(w) = 0')
problem.add_equation('bz - dz(b) = 0')
problem.add_bc('left(u) = 0')
problem.add_bc('left(v) = 0')
problem.add_bc('left(w) = 0')
problem.add_bc('left(bz) = 0')
problem.add_bc('right(uz) = 0')
problem.add_bc('right(vz) = 0')
problem.add_bc('right(w) = 0')
problem.add_bc('right(bz) = 0')

solver = problem.build_solver()

# Create function to compute max growth rate for given kx
def max_growth_rate(ly):
    logger.info('Computing max growth rate for ly = %f' %ly)
    # Change kx parameter
    problem.namespace['l'].value = ly
    problem.namespace['k'].value = 0 # for now only considering baroclinic axis
    # Solve for eigenvalues with sparse search near zero, rebuilding NCCs
#    solver.solve_sparse(solver.pencils[0], N=10, target=0, rebuild_coeffs=True)
    solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)
    omg = solver.eigenvalues
    omg[np.isnan(omg)] = 0.
    omg[np.isinf(omg)] = 0.
    idx = np.argsort(omg.imag)
#        print(str(idx[-1]))
    # Return largest imaginary part
    return omg[idx[-1]].imag

# Compute growth rate over local wavenumbers
ly_local = ly_global[CW.rank::CW.size]

t1 = time.time()
growth_local = np.array([max_growth_rate(ly) for ly in ly_local])
t2 = time.time()
logger.info('Elapsed solve time: %f' %(t2-t1))

# Reduce growth rates to root process
growth_global = np.zeros_like(ly_global)
growth_global[CW.rank::CW.size] = growth_local
if CW.rank == 0:
    CW.Reduce(MPI.IN_PLACE, growth_global, op=MPI.SUM, root=0)
else:
    CW.Reduce(growth_global, growth_global, op=MPI.SUM, root=0)

# Plot growth rates from root process
if CW.rank == 0:
    name = 'StabilityData_'+str(Ri) # Can vary this depending on parameter of interest
    np.savez(directoryname+name + '.npz', nz=nz, tht=tht, z=z1, f=f,
    V=V['g'], B=B['g'], Bz=Bz['g'], Vz=Vz['g'], Lz = Lz, ll=ly_global,N=N,
    gr=growth_global)
