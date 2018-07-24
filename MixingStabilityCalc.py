import numpy as np
#import matplotlib.pyplot as plt
from mpi4py import MPI
CW = MPI.COMM_WORLD
from dedalus import public as de
import time

import logging
logger = logging.getLogger(__name__)

# parameters
N = 1e-3 # buoyancy frequency
f = -5.5e-5 # Coriolis parameter
tht = 2e-3 # slope angle
kap_0 = 1e-5 # background diffusivity
kap_1 = 1e-3 # bottom enhancement of diffusivity
h = 200. # decay scale of mixing
Pr = 1 # Prandtl number
H = 2500. # domain height

# along-slope wavenumbers
ll_global = np.logspace(-5, -2, 192)

# number of grid points
nz = 256

# file name that results are saved in
directory = '/home/jacob/dedalus/UpdatedMixingStability/'

# build domain
z_basis = de.Chebyshev('z', nz, interval=(0, H))
domain = de.Domain([z_basis], np.complex128, comm=MPI.COMM_SELF)

# non-constant coefficients
kap = domain.new_field(name='kap')
z = domain.grid(0)
kap['g'] = kap_0 + kap_1*np.exp(-z/h)

# STEADY STATE

# setup problem
problem = de.LBVP(domain, variables=['U', 'V', 'B', 'Uz', 'Vz', 'Bz'])
problem.parameters['N'] = N
problem.parameters['f'] = f
problem.parameters['tht'] = tht
problem.parameters['kap'] = kap
problem.parameters['Pr'] = Pr
problem.add_equation(('-f*V*cos(tht) - B*sin(tht) - Pr*(dz(kap)*Uz'
        '+ kap*dz(Uz)) = 0'))
problem.add_equation('f*U*cos(tht) - Pr*(dz(kap)*Vz + kap*dz(Vz)) = 0')
problem.add_equation(('U*N**2*sin(tht) - dz(kap)*Bz - kap*dz(Bz)'
        '= dz(kap)*N**2*cos(tht)'))
problem.add_equation('Uz - dz(U) = 0')
problem.add_equation('Vz - dz(V) = 0')
problem.add_equation('Bz - dz(B) = 0')
problem.add_bc('left(U) = 0')
problem.add_bc('left(V) = 0')
problem.add_bc('left(Bz) = -N**2*cos(tht)')
problem.add_bc('right(Uz) = 0')
problem.add_bc('right(Vz) = 0')
problem.add_bc('right(Bz) = 0')

# build solver and solve
solver = problem.build_solver()
solver.solve()

# collect solution
U = solver.state['U']
V = solver.state['V']
B = solver.state['B']
Uz = solver.state['Uz']
Vz = solver.state['Vz']
Bz = solver.state['Bz']

# LINEAR STABILITY

problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p', 'uz', 'vz', 'wz',
        'bz'], eigenvalue='omg')
problem.parameters['U'] = U
problem.parameters['V'] = V
problem.parameters['B'] = B
problem.parameters['Uz'] = Uz
problem.parameters['Vz'] = Vz
problem.parameters['Bz'] = Bz
problem.parameters['N'] = N
problem.parameters['f'] = f
problem.parameters['tht'] = tht
problem.parameters['kap'] = kap
problem.parameters['Pr'] = Pr
problem.parameters['k'] = 0. # will be set in loop
problem.parameters['l'] = 0. # will be set in loop
problem.substitutions['dx(A)'] = "1j*k*A"
problem.substitutions['dy(A)'] = "1j*l*A"
problem.substitutions['dt(A)'] = "-1j*omg*A"
problem.add_equation(('dt(u) + U*dx(u) + V*dy(u) + w*Uz - f*v*cos(tht) + dx(p)'
        '- b*sin(tht) - Pr*(kap*dx(dx(u)) + kap*dy(dy(u)) + dz(kap)*uz'
        '+ kap*dz(uz)) = 0'))
problem.add_equation(('dt(v) + U*dx(v) + V*dy(v) + w*Vz + f*u*cos(tht)'
        '- f*w*sin(tht) + dy(p) - Pr*(kap*dx(dx(v)) + kap*dy(dy(v))'
        '+ dz(kap)*vz + kap*dz(vz)) = 0'))
problem.add_equation(('dt(w) + U*dx(w) + V*dy(w) + f*v*sin(tht) + dz(p)'
        '- b*cos(tht) - Pr*(kap*dx(dx(w)) + kap*dy(dy(w)) + dz(kap)*wz'
        '+ kap*dz(wz)) = 0'))
problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*N**2*sin(tht)'
        '+ w*(N**2*cos(tht) + Bz) - kap*dx(dx(b)) - kap*dy(dy(b)) - dz(kap)*bz'
        '- kap*dz(bz) = 0'))
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
#problem.add_bc('right(w) = 0')
problem.add_bc('right(w) = -right(u)*tan(tht)')
problem.add_bc('right(bz) = 0')

# set up solver
solver = problem.build_solver()

def sorted_eigen(k, l):

    """
    Solves eigenvalue problem and returns sorted eigenvalues and associated
    eigenvectors.
    """

    # set wavenumbers
    problem.namespace['k'].value = k
    problem.namespace['l'].value = l

    # solve problem
    solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)

    # sort eigenvalues
    omg = solver.eigenvalues
    omg[np.isnan(omg)] = 0.
    omg[np.isinf(omg)] = 0.
    idx = np.argsort(omg.imag)

    return idx

def max_growth_rate(k, l):

    """Finds maximum growth rate for given wavenumbers k, l."""

    print(k, l)

    # solve eigenvalue problem and sort
    idx = sorted_eigen(k, l)

    return solver.eigenvalues[idx[-1]].imag


    # Compute growth rate over local wavenumbers
ly_local = ll_global[CW.rank::CW.size]
    
t1 = time.time()
growth_local = np.array([max_growth_rate(0, ly) for ly in ly_local])
t2 = time.time()
logger.info('Elapsed solve time: %f' %(t2-t1))

# Reduce growth rates to root process
growth_global = np.zeros_like(ll_global)
growth_global[CW.rank::CW.size] = growth_local
if CW.rank == 0:
    CW.Reduce(MPI.IN_PLACE, growth_global, op=MPI.SUM, root=0)
else:
    CW.Reduce(growth_global, growth_global, op=MPI.SUM, root=0)

# Plot growth rates from root process
if CW.rank == 0:    

    # get full eigenvectors and eigenvalues for l with largest growth
    idx = sorted_eigen(0., ll_global[np.argmax(growth_global)])
    solver.set_state(idx[-1])

    # collect eigenvector
    u = solver.state['u']
    v = solver.state['v']
    w = solver.state['w']
    b = solver.state['b']
    uz = solver.state['uz']
    vz = solver.state['vz']
    
    # shear production
    SP = -2*np.real(np.conj(w['g'])*(u['g']*Uz['g']+v['g']*Vz['g']))
    
    # buoyancy production
    BP = 2*np.real((u['g']*np.sin(tht)+w['g']*np.cos(tht))*np.conj(b['g']))
    
    DISS = -kap['g']*(np.abs(uz['g'])**2 + np.abs(vz['g'])**2)
    # SAVE TO FILE
    #
    np.savez(directory + 'MixingStabilityOut' + '.npz', nz=nz, N=N, tht=tht, z=z, kap=kap['g'], Pr=Pr, U=U['g'], Uz=Uz['g'], Vz=Vz['g'],
            V=V['g'], B=B['g'], Bz=Bz['g'], u=u['g'], v=v['g'], w=w['g'], b=b['g'], ll=ll_global,
            gr=growth_global, SP=SP, BP=BP, DISS=DISS)

