#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:14:37 2017

@author: jacob
"""
from linearStability import linearStabilityAnalysis
import numpy as np
from dedalus import public as de
import scipy.integrate as integrate
from pylab import *
from joblib import Parallel, delayed
import multiprocessing
import time

#from linearStability import ls

directoryname = "./SlopeAngle/"

# Physical parameters
f = 1e-4
tht = 0
Pr = 1
H = 2000
thtarr = [0, 1e-3, 1e-2]

# Grid Parameters
nz = 256
ll = np.logspace(-1, 1, 32)*f/.1

# Define domain
z_basis = de.Chebyshev('z', nz, interval=(0, H))
domain = de.Domain([z_basis], np.complex128)
z = domain.grid(0)

t0 = time.time()
# Define Stability Analysis Parameters
kap = domain.new_field(name='kap')
kap['g'] = 0*z
U = domain.new_field(name='U')
U['g'] = 0*z
Uz = domain.new_field(name='Uz')
Uz['g'] = 0*z
V = domain.new_field(name='V')
V['g'] = 0.0001*(z)
Vz = domain.new_field(name='Vz')
Vz['g'] = 0.0001*(z-z+1)
Ri = 100
Bzf = Ri*Vz['g']**2
tpoint = np.floor( next((x[0] for x in enumerate(z) if x[1]>250)))
Bstr  = -0.5*(np.tanh((-z + z[tpoint])/40)+1)
Bzt = Bzf*10**(2*Bstr)
Bt = np.zeros([nz])
Bt[1:nz] = integrate.cumtrapz(Bzt, z)
B = domain.new_field(name='B')
B['g'] = Bt
Bz = domain.new_field(name='Bz')
Bz['g'] = Bzt

def parprocess(tht):
#    tht = thtarr[i]
    args = (f, tht, kap, Pr, U, Uz, V, Vz, B, Bz, nz, H, ll, domain)
    (gr, u, v, w, b, SP, BP) = linearStabilityAnalysis(*args)

    name = 'StabilityData_'+str(tht) # Can vary this depending on parameter of interest

    np.savez(directoryname+name + '.npz', nz=nz, tht=tht, z=z, f=f, kap=kap['g'], Pr=Pr, U=U['g'],
        V=V['g'], B=B['g'], u=u['g'], v=v['g'], w=w['g'], b=b['g'], ll=ll,
        gr=gr, SP=SP, BP=BP)

numcores = 8
Parallel(n_jobs = numcores)(delayed(parprocess)(i) for i in thtarr)

#for i in range(size(thtarr)):
    
t1 = time.time()
print('Complete - Normal, elapsed time: '+str(t1-t0))