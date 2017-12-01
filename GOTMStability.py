#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:50:30 2017

@author: jacob
"""
import time

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import logging
logger = logging.getLogger(__name__)

pathtofile = '/data/thomas/jacob13/STABILITY/GOTMOUT/entrainment_down.nc'
pathtosave = '/data/thomas/jacob13/STABILITY/GOTMEkmanD/'

nc_fid = Dataset(pathtofile, 'r')  # Dataset is the class behavior to open the file

bg = np.squeeze(nc_fid.variables['buoy'][:])
ug = np.squeeze(nc_fid.variables['u'][:])
vg = np.squeeze(nc_fid.variables['v'][:])
kappag = np.squeeze(nc_fid.variables['nuh'][:]) + 1e-5
nug = np.squeeze(nc_fid.variables['num'][:]) + 1e-5
nng = np.squeeze(nc_fid.variables['NN'][:])
ntg, nzg = vg.shape
zg = np.linspace(0.25, 147.75, 300)
ts = 3600

# STABILITY PARAMETERS
f = 1e-4
tht = 1e-2
Nb = (3.5e-3)

H = 150
nz = 256



ly_global = np.linspace(1e-4, 1e-2, 192)

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
z_basis = de.Chebyshev('z', nz, interval=(0,H))
domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

z = domain.grid(0)

# Define Stability Analysis Parameters

kap = domain.new_field(name='kap')
nu = domain.new_field(name='nu')
U = domain.new_field(name='U')
Uz = domain.new_field(name='Uz')
#Uz['g'] = 0*z
V = domain.new_field(name='V')
Vz = domain.new_field(name='Vz')
Bz = domain.new_field(name='Bz')
B = domain.new_field(name='B')

for i in range(0, ntg, 6):
    
    problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p', 'uz', 'vz', 'wz',
            'bz'], eigenvalue='omg', tolerance = 1e-10)
    kap['g'] = np.interp(z,zg, kappag[i,1:])
    nu['g'] = np.interp(z, zg, nug[i,1:])
    U['g'] = np.interp(z,zg, ug[i,:])
    box = np.ones(5)/5
    kap['g'] = np.convolve(kap['g'], box, mode='same')
    nu['g'] = np.convolve(nu['g'], box, mode='same')
    U['g'] = np.convolve(U['g'], box, mode='same')
    Uz  = U.differentiate(z_basis)
    V['g'] = np.interp(z, zg, vg[i,:])
    V['g'] = np.convolve(V['g'], box, mode='same')
    Vz = V.differentiate(z_basis)
    B['g'] = np.interp(z, zg, bg[i,:])
    B['g'] = np.convolve(B['g'], box, mode='same')
    Bz = B.differentiate(z_basis)
    Bz['g'] = np.interp(z, zg[0:-10], nng[i,0:-10])
    
    Bt = np.zeros([nz])
    Bt[1:nz] = integrate.cumtrapz(Bz['g'], z)
    B['g'] = Bt - Bt[-1]
    
    problem.parameters['tht'] = tht
    problem.parameters['U'] = U
    problem.parameters['V'] = V
    problem.parameters['B'] = B
    problem.parameters['Uz'] = Uz
    problem.parameters['Vz'] = Vz
    problem.parameters['Bz'] = Bz
    problem.parameters['N'] =Nb
    problem.parameters['f'] = f
    problem.parameters['kap'] = kap
    problem.parameters['nu'] = nu
    problem.parameters['k'] = 0. # will be set in loop
    problem.parameters['l'] = 0. # will be set in loop
    problem.substitutions['dx(A)'] = "1j*k*A"
    problem.substitutions['dy(A)'] = "1j*l*A"
    problem.substitutions['dt(A)'] = "-1j*omg*A"
    problem.add_equation(('dt(u) + U*dx(u) + V*dy(u) + w*Uz - f*v*cos(tht) + dx(p)'
            '- b*sin(tht) - (dz(nu)*uz'
            '+ nu*dz(uz)) = 0'))
    problem.add_equation(('dt(v) + U*dx(v) + V*dy(v) + w*Vz + f*u*cos(tht)'
            '- f*w*sin(tht) + dy(p) - (dz(nu)*vz + nu*dz(vz)) = 0'))
    problem.add_equation(('(dt(w) + U*dx(w) + V*dy(w)) + f*v*sin(tht) + dz(p)'
            '- b*cos(tht) - (dz(nu)*wz + nu*dz(wz)) = 0'))
    problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*N**2*sin(tht)'
                '+ w*Bz - dz(kap)*bz - kap*dz(bz) = 0'))
    problem.add_equation('dx(u) + dy(v) + wz = 0')
    problem.add_equation('uz - dz(u) = 0')
    problem.add_equation('vz - dz(v) = 0')
    problem.add_equation('wz - dz(w) = 0')
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
    def sorted_eigen(ky, ly):
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
        return idx
    
    def max_growth_rate(l):

        """Finds maximum growth rate for given wavenumbers k, l."""
        k = 0
        #print(k, l)

        # solve eigenvalue problem and sort
        idx = sorted_eigen(k, l)

        return solver.eigenvalues[idx[-1]].imag
    
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
        # get full eigenvectors and eigenvalues for l with largest growth
        idx = sorted_eigen(0., ly_global[np.argmax(growth_global)])
        solver.set_state(idx[-1])

        # collect eigenvector
        up = solver.state['u']
        vp = solver.state['v']
        wp = solver.state['w']
        bp = solver.state['b']
        
        uzp = solver.state['uz']
        vzp = solver.state['vz']
        
        # shear production
        SP = -2*np.real(np.conj(wp['g'])*(up['g']*Uz['g']+vp['g']*Vz['g']))

        # buoyancy production
        BP = 2*np.real((up['g']*np.sin(tht)+wp['g']*np.cos(tht))*np.conj(bp['g']))
        
        DISS = -2*np.real(nu['g']*(np.conj(uzp['g'])*uzp['g'] + np.conj(vzp['g'])*vzp['g']))
        
        name = 'StabilityData_'+str(i) # Can vary this depending on parameter of interest
        np.savez(pathtosave+name, nz=nz, tht=tht, z=z, f=f, kap=kap['g'], nu=nu['g'], U=U['g'],
        V=V['g'], B=B['g'], Bz=Bz['g'], N = Nb, Vz=Vz['g'], H = H, ll=ly_global, time=i*ts,
        gr=growth_global, u=up['g'], v=vp['g'], w = wp['g'], b = bp['g'], SP = SP, BP=BP, DISS=DISS)
        
