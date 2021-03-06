"""
Stone-on-a-slope

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

directoryname = "/home/jacob/dedalus/MixingStabilityIdeal/" # Where to save the output


# Physical parameters
f = -5.5e-5

tht = 0
Pr = 1

H = 500
Ri = 25

Shmag = .015/H
Bzmag = Shmag**2*Ri

#thtarr = np.linspace(-2, 2, 256)*Shmag*f/Bzmag
thtarr = [2e-3]
# Grid Parameters
nz = 256

#ly_global = np.linspace(1e-2, 4.25, 256)*f/(np.sqrt(Bzmag)*H)
ly_global = np.logspace(-5, -2, 192)


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
Bz = domain.new_field(name='Bz')
B = domain.new_field(name='B')
V['g'] = Shmag*(z)
Vz['g'] = Shmag*(z-z+1) #Note this assumes no horizotal variation (ie. won't work for the non-uniform case)
Bt = np.zeros([nz])
Bz['g'] = np.array(Bzmag*np.ones([nz]))
Bt[1:nz] = integrate.cumtrapz(Bz['g'], z)
B['g'] = Bt

# 2D Boussinesq hydrodynamics, with no-slip boundary conditions
# Use substitutions for x and t derivatives
for tht in thtarr:
#    bz = 1/(2*Ri*np.sin(tht)**2)*(f**2*np.cos(tht)+np.sqrt(f**4*np.cos(tht)**2 + 4*Bzmag*f**2*Ri*np.sin(tht)**2))
#    Shmag = -bz/(f*np.sin(tht))

    V['g'] = Shmag*(z)/np.cos(tht)

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
        
        name = 'StabilityData_'+str(tht) # Can vary this depending on parameter of interest
        np.savez(directoryname+name + '.npz', nz=nz, tht=tht, z=z, f=f, Pr=Pr, U=U['g'],
        V=V['g'], B=B['g'], Bz=Bz['g'], Vz=Vz['g'], H = H, ll=ly_global,
        gr=growth_global)
