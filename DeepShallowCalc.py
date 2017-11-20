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
directoryname = "/home/jacob/dedalus/DeepShallowModes/"

# Physical parameters
f = 1e-4
tht = 0
Pr = 1
H = 1000
BLH = 250
Ri = 100
RiBL = 1
#Bzmag = 2.5e-5
#Shmag = np.sqrt(Bzmag/Ri)
Shmag = .1/H
Bzmag = Ri*Shmag**2
BzmagBL = RiBL*Shmag**2

#Shmag = 0;
thtarr = np.linspace(0, 1e-2, 5)

#Ri = 
#Shmag = 1e-4
#Bzmag = (Shmag/Ro)**2 # Ro = Uz/N
# Grid Parameters
nz = 128#256

ly_global = np.linspace(1e-4, 1e-2, 64)
ly_global = np.logspace(-4, -2, 128)
# Create bases and domain
# Use COMM_SELF so keep calculations independent between processes
z_basis = de.Chebyshev('z', nz, interval=(0,H))
domain = de.Domain([z_basis], grid_dtype=np.complex128, comm=MPI.COMM_SELF)

z = domain.grid(0)

# Define Stability Analysis Parameters

kap = domain.new_field(name='kap')
kap['g'] = 1e-5*np.ones(z.shape)
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
zind = np.floor( next((x[0] for x in enumerate(z) if x[1]>BLH)))
#Bz['g'][0:zind] = BzmagBL

tpoint = np.floor( next((x[0] for x in enumerate(z) if x[1]>BLH)))
Bstr  = -0.5*(np.tanh((-z + z[tpoint])/40)+1)
Bz['g'] = Bz['g']*10**(2*Bstr)
    
    
Bt[1:nz] = integrate.cumtrapz(Bz['g'], z)
B['g'] = Bt
#%%
# 2D Boussinesq hydrodynamics, with no-slip boundary conditions
# Use substitutions for x and t derivatives
for tht in thtarr:
#    bz = 1/(2*Ri*np.sin(tht)**2)*(f**2*np.cos(tht)+np.sqrt(f**4*np.cos(tht)**2 + 4*Bzmag*f**2*Ri*np.sin(tht)**2))
#    Shmag = -bz/(f*np.sin(tht))
    
    V['g'] = Shmag*(z)/np.cos(tht)

    problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p'], eigenvalue='omg', tolerance = 1e-12)
    problem.parameters['tht'] = tht
    problem.parameters['V'] = V
    problem.parameters['Uz'] = Uz
    problem.parameters['Vz'] = Vz
    problem.parameters['NS'] = Bz
    problem.parameters['f'] = f
    problem.parameters['tht'] = tht
    problem.parameters['kap'] = kap
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
    problem.add_bc('right(w) = -u*tan(tht)')


#    problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p', 'uz', 'vz', 'wz',
#            'bz'], eigenvalue='omg', tolerance = 1e-10)
#    problem.parameters['tht'] = tht
#    problem.parameters['U'] = U
#    problem.parameters['V'] = V
#    problem.parameters['B'] = B
#    problem.parameters['Uz'] = Uz
#    problem.parameters['Vz'] = Vz
#    problem.parameters['NS'] = Bz
#    problem.parameters['f'] = f
#    problem.parameters['tht'] = tht
#    problem.parameters['kap'] = kap
#    problem.parameters['Pr'] = Pr
#    problem.parameters['k'] = 0. # will be set in loop
#    problem.parameters['l'] = 0. # will be set in loop
#    problem.substitutions['dx(A)'] = "1j*k*A"
#    problem.substitutions['dy(A)'] = "1j*l*A"
#    problem.substitutions['dt(A)'] = "-1j*omg*A"
#    problem.add_equation(('dt(u) + U*dx(u) + V*dy(u) + w*Uz - f*v*cos(tht) + dx(p)'
#            '- b*sin(tht) - Pr*(kap*dx(dx(u)) + kap*dy(dy(u)) + dz(kap)*uz'
#            '+ kap*dz(uz)) = 0'))
#    problem.add_equation(('dt(v) + U*dx(v) + V*dy(v) + w*Vz*cos(tht) + f*u*cos(tht)'
#            '- f*w*sin(tht) + dy(p) - Pr*(kap*dx(dx(v)) + kap*dy(dy(v))'
#            '+ dz(kap)*vz + kap*dz(vz)) = 0'))
#    problem.add_equation(('(dt(w) + U*dx(w) + V*dy(w)) + f*v*sin(tht) + dz(p)'
#            '- b*cos(tht) - Pr*(kap*dx(dx(w)) + kap*dy(dy(w)) + dz(kap)*wz'
#            '+ kap*dz(wz)) = 0'))
##    problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*(NS*sin(tht))'
##                '+ w*(NS*cos(tht)) - kap*dx(dx(b)) - kap*dy(dy(b)) - dz(kap)*bz'
##                '- kap*dz(bz) = 0'))
#    problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*(NS*sin(tht)+f*Vz*cos(tht))'
#            '+ w*(NS*cos(tht)-f*Vz*sin(tht)) - kap*dx(dx(b)) - kap*dy(dy(b)) - dz(kap)*bz'
#            '- kap*dz(bz) = 0'))
#    #problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*Vz*f'
#    #        '+ w*(Bz) - kap*dx(dx(b)) - kap*dy(dy(b)) - dz(kap)*bz'
#    #        '- kap*dz(bz) = 0'))
#    problem.add_equation('dx(u) + dy(v) + wz = 0')
#    problem.add_equation('uz - dz(u) = 0')
#    problem.add_equation('vz - dz(v) = 0')
#    problem.add_equation('wz - dz(w) = 0')
#    problem.add_equation('bz - dz(b) = 0')
#    problem.add_bc('left(u) = 0')
#    problem.add_bc('left(v) = 0')
#    problem.add_bc('left(w) = 0')
#    problem.add_bc('left(bz) = 0')
#    problem.add_bc('right(uz) = 0')
#    problem.add_bc('right(vz) = 0')
#    problem.add_bc('right(w) = -right(u)*tan(tht)') # Flat upper boundary
##    problem.add_bc('right(w) = 1/10*(dt(right(p)) + right(u)*dx(right(p))+right(v)*dy(right(p)))')
#    problem.add_bc('right(bz) = 0')
    
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
#        plt.plot(ly_global*np.sqrt(Bz['g'][-1])*H/f, growth_global*np.sqrt(Bz['g'][-1])/(f*Vz['g'][-1]), '.')
#        plt.ylim((0, 1))
#        plt.xlabel(r'$ly$')
#        plt.ylabel(r'$\mathrm{Im}(\omega)$')
#        plt.title('Growth Rates')
#        plt.savefig('growth_rates_%.4f_%.1f.png' %(tht, nz))
        
        name = 'StabilityData_'+str(tht) # Can vary this depending on parameter of interest
        np.savez(directoryname+name + '.npz', nz=nz, tht=tht, z=z, f=f, kap=kap['g'], Pr=Pr, U=U['g'],
        V=V['g'], B=B['g'], Bz=Bz['g'], Vz=Vz['g'], H = H, ll=ly_global,
        gr=growth_global)
