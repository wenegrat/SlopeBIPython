"""
Dedalus script for calculating the maximum growth rates in no-slip
Rayleigh Benard convection over a range of horizontal wavenumbers.

This script can be ran serially or in parallel, and produces a plot of the
highest growth rate found for each horizontal wavenumber.

To run using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
from mpi4py import MPI
CW = MPI.COMM_WORLD
from pylab import *
import scipy.integrate as integrate

import logging
logger = logging.getLogger(__name__)


# Global parameters
directoryname = "/data/thomas/jacob13/STABILITY/SlopeAngleRiVar100/"

# Physical parameters
f = 1e-4
tht = 0
Pr = 1
H = 100
Ri = 100
# FIX Vz and Slope Angle
tht = 5e-3
Shmag = 0.1/H

# Richardson numbers to scan
Riv = np.linspace(0.0, 10, 256)

nz = 64

ly_global = np.linspace(1e-4, 3, 256)*f/(Shmag*H)

# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
z_basis = de.Chebyshev('z', nz, interval=(0,H))
domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

z = domain.grid(0)

# Define Stability Analysis Parameters

U = domain.new_field(name='U')
U['g'] = 0*z
Uz = domain.new_field(name='Uz')
Uz['g'] = 0*z
V = domain.new_field(name='V')
Vz = domain.new_field(name='Vz')
V['g'] = Shmag*(z)/np.cos(tht)
Vz['g'] = Shmag*(z-z+1) 
Bz = domain.new_field(name='Bz')
Bz['g'] = np.array(np.ones([nz]))


# 2D Boussinesq hydrodynamics, with no-slip boundary conditions
# Use substitutions for x and t derivatives
for Ri in Riv:

    Bzmag = Ri*Shmag**2 
    Bz['g'] = np.array(Bzmag*np.ones([nz]))


    problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p'], eigenvalue='omg', tolerance = 1e-10)
    problem.parameters['tht'] = tht
    problem.parameters['V'] = V
    problem.parameters['Uz'] = Uz
    problem.parameters['Vz'] = Vz
    problem.parameters['NS'] = Bz
    problem.parameters['f'] = f
    problem.parameters['tht'] = tht
    problem.parameters['Pr'] = Pr
    problem.parameters['k'] = 0. # will be set in loop
    problem.parameters['l'] = 0. # will be set in loop
    problem.substitutions['dx(A)'] = "1j*k*A"
    problem.substitutions['dy(A)'] = "1j*l*A"
    problem.substitutions['dt(A)'] = "-1j*omg*A"
    problem.add_equation(('dt(u) + V*dy(u) - f*v*cos(tht) + dx(p)- b*sin(tht) = 0'))
    problem.add_equation(('dt(v) + V*dy(v) + w*Vz/cos(tht) + f*u*cos(tht)- f*w*sin(tht) + dy(p) = 0'))
    problem.add_equation(('(dt(w) + V*dy(w)) + f*v*sin(tht) + dz(p)- b*cos(tht) = 0'))
    problem.add_equation(('dt(b) +  V*dy(b) + u*(NS*sin(tht)+f*Vz*cos(tht))'
            '+ w*(NS*cos(tht)-f*Vz*sin(tht)) = 0'))
    problem.add_equation('dx(u) + dy(v) + dz(w) = 0')

    problem.add_bc('left(w) = 0')
    problem.add_bc('right(w) = 0')

    
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
    
    # Save growth rates from root process
    if CW.rank == 0:

        name = 'StabilityData_'+str(Ri) # Can vary this depending on parameter of interest
        np.savez(directoryname+name + '.npz', nz=nz, tht=tht, z=z, f=f, Pr=Pr, U=U['g'],
        V=V['g'], Bz=Bz['g'], Vz=Vz['g'], H = H, Ri = Ri, ll=ly_global,
        gr=growth_global)
