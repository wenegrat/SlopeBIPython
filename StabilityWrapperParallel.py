#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:14:37 2017

@author: jacob
"""
from linearStabilityParallel import linearStabilityAnalysisPar
import numpy as np
from dedalus import public as de
import scipy.integrate as integrate
from pylab import *
from joblib import Parallel, delayed
import multiprocessing
import time
import sys
sys.path.append("/home/jacob/dedalus/eigenvals/")
t0 = time.time()
# Program parameters
directoryname = "./SlopeAngle/"
numcores = 20

# Physical parameters
f = 1e-4
tht = 0
Pr = 1
H = 100
Ri = 1e0
Bzmag = 1e-7
Shmag = np.sqrt(Bzmag)/Ri
thtarr = np.linspace(-1.5, 1.5, 20)*Shmag*f/Bzmag

#Ri = 
#Shmag = 1e-4
#Bzmag = (Shmag/Ro)**2 # Ro = Uz/N
# Grid Parameters
nz = 32#128#256
ll = np.logspace(-1, 1, 64)*f/.1
ll = np.logspace(-4, -1, 128)
ll = np.linspace(0, 3, 32)*f/(np.sqrt(Bzmag)*H)

# Define domain
z_basis = de.Chebyshev('z', nz, interval=(0, H))
#z_basis = de.SinCos('z', nz, interval=(0, H))

domain = de.Domain([z_basis], np.complex128)
z = domain.grid(0)

# Define Stability Analysis Parameters
kap = domain.new_field(name='kap')
kap['g'] = 0*np.ones(z.shape)
U = domain.new_field(name='U')
U['g'] = 0*z
Uz = domain.new_field(name='Uz')
Uz['g'] = 0*z
#V = domain.new_field(name='V')
#V['g'] = 0.0001*(z)
#Vz = domain.new_field(name='Vz')
#Vz['g'] = 0.0001*(z-z+1)
#Ri = 100
#Bzf = Ri*Vz['g']**2
#tpoint = np.floor( next((x[0] for x in enumerate(z) if x[1]>400)))
#Bstr  = -0.5*(np.tanh((-z + z[tpoint])/80)+1)
#Bzt = Bzf*10**(2*Bstr)
#Bt = np.zeros([nz])
#Bt[1:nz] = integrate.cumtrapz(Bzt, z)
#B = domain.new_field(name='B')
#B['g'] = Bt
#Bz = domain.new_field(name='Bz')
#Bz['g'] = Bzt

# Stability Analysis QG limit


#B['g'] = integrate.cumtrapz(Bzmag*np.ones([nz]), z)



def parprocess(l):
#    tht = thtarr[i]
    args = (f, tht, kap, Pr, U, Uz, V, Vz, B, Bz, nz, H, l,0, domain)
    (grp, u, v, w, b) = linearStabilityAnalysisPar(*args)
    return grp

V = domain.new_field(name='V')
Vz = domain.new_field(name='Vz')
Bz = domain.new_field(name='Bz')
B = domain.new_field(name='B')

# Iterate over Slope Angles (could modify for any parameter)
for tht in thtarr:
    progress = 'Processing Theta: {:3.3f}  Elapsed Time (hours): {:3.3f}'.format(tht, (time.time()-t0)/3600)
    print(progress)
    V['g'] = Shmag*(z)
    Vz['g'] = Shmag*(z-z+1)*cos(tht) #Note this assumes no horizotal variation (ie. won't work for the non-uniform case)
    Bt = np.zeros([nz])
    Bz['g'] = Bzmag*np.ones([nz])*cos(tht) - Shmag*f*sin(tht)
    Bt[1:nz] = integrate.cumtrapz(Bz['g'], z)
    B['g'] = Bt

    
    gr = Parallel(n_jobs = numcores)(delayed(parprocess)(l) for l in ll) # Get the growth rates
#    idx = np.argsort(gr) 
    
    # Add one additional call to get the structure of the most unstable mode
    args = (f, tht, kap, Pr, U, Uz, V, Vz, B, Bz, nz, H, ll[np.argmax(gr)],0, domain)
    (grp, u, v, w, b) = linearStabilityAnalysisPar(*args)
    
    # Shear production
    SP = -2*np.real(np.conj(w)*(u*Uz['g']+v*Vz['g']))

    # Buoyancy production
    BP = 2*np.real((u*np.sin(tht)+w*np.cos(tht))*np.conj(b))

    # Save variables    
    name = 'StabilityData_'+str(tht) # Can vary this depending on parameter of interest
    np.savez(directoryname+name + '.npz', nz=nz, tht=tht, z=z, f=f, kap=kap['g'], Pr=Pr, U=U['g'],
        V=V['g'], B=B['g'], Bz=Bz['g'], Vz=Vz['g'], H = H, u=u, v=v, w=w, b=b, ll=ll,
        gr=gr, SP=SP, BP=BP)
    
t1 = time.time()
print('Complete - Normal, elapsed time: '+str((t1-t0)/3600))