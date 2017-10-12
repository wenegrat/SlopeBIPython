import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import scipy.integrate as integrate
from dedalus import public as de

import logging
logger = logging.getLogger(__name__)

# parameters
#N = 1e-3 # buoyancy frequency
f = 2.5e-5 # Coriolis parameter
f = 1e-4
thtarr = [0, 0.0025, 0.0075, 0.01] # [1e-3, 2e-3, 4e-3] # slope angle
kap_0 = 0 #1e-5 # background diffusivity
kap_1 = 0 #1e-3 # bottom enhancement of diffusivity
h = 200. # decay scale of mixing
Pr = 1 # Prandtl number
H = 1000. # domain height

# along-slope wavenumbers
ll = np.logspace(-1, 1, 128)*f/.1

# number of grid points
nz = 128

# file name that results are saved in
name = 'test'
#%%
# build domain
z_basis = de.Chebyshev('z', nz, interval=(0, H))
domain = de.Domain([z_basis], np.complex128)

# non-constant coefficients
kap = domain.new_field(name='kap')
z = domain.grid(0)
kap['g'] = kap_0 + kap_1*np.exp(-z/h)

grt = zeros([size(thtarr), size(ll)])
# STEADY STATE
for i in range(size(thtarr)):
    tht = thtarr[i]
    # setup problem
#    problem = de.LBVP(domain, variables=['U', 'V', 'B', 'Uz', 'Vz', 'Bz'])
#    problem.parameters['N'] = N
#    problem.parameters['f'] = f
#    problem.parameters['tht'] = tht
#    problem.parameters['kap'] = kap
#    problem.parameters['Pr'] = Pr
#    problem.add_equation(('-f*V*cos(tht) - B*sin(tht) - Pr*(dz(kap)*Uz'
#            '+ kap*dz(Uz)) = 0'))
#    problem.add_equation('f*U*cos(tht) - Pr*(dz(kap)*Vz + kap*dz(Vz)) = 0')
#    problem.add_equation(('U*N**2*sin(tht) - dz(kap)*Bz - kap*dz(Bz)'
#            '= dz(kap)*N**2*cos(tht)'))
#    problem.add_equation('Uz - dz(U) = 0')
#    problem.add_equation('Vz - dz(V) = 0')
#    problem.add_equation('Bz - dz(B) = 0')
#    problem.add_bc('left(U) = 0')
#    problem.add_bc('left(V) = 0')
#    problem.add_bc('left(Bz) = -N**2*cos(tht)')
#    problem.add_bc('right(Uz) = 0')
#    problem.add_bc('right(Vz) = 0')
#    problem.add_bc('right(Bz) = 0')
#    
#    # build solver and solve
#    solver = problem.build_solver()
#    solver.solve()
#    
#    # collect solution
#    U = solver.state['U']
#    V = solver.state['V']
#    B = solver.state['B']
#    Uz = solver.state['Uz']
#    Vz = solver.state['Vz']
#    Bz = solver.state['Bz']
    
    # Use Ad Hoc Structure
# non-constant coefficients
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
    
    #%%
#    V = 0.0001*np.ones([nz])*(z/H)
#    Vz = 0.0001*np.ones([nz])
#    U = 0*V
#    Uz = 0*Vz
#    Ri = 100
#    Bzf = Ri*Vz**2
#    tpoint = np.floor( next((x[0] for x in enumerate(z) if x[1]>250)))
#    Bstr  = -0.5*(np.tanh((-z + z[tpoint])/40)-1)
#    Bzt = Bzf*10**(2*Bstr)
#    Bt = np.zeros([nz])
#    Bt[1:nz] = integrate.cumtrapz(Bz, z)
    
    
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
            '- b*sin(tht) - Pr*(kap*dx(dx(u)) + dy(dy(u)) + dz(kap)*uz'
            '+ kap*dz(uz)) = 0'))
    problem.add_equation(('dt(v) + U*dx(v) + V*dy(v) + w*Vz + f*u*cos(tht)'
            '- f*w*sin(tht) + dy(p) - Pr*(kap*dx(dx(v)) + kap*dy(dy(v))'
            '+ dz(kap)*vz + kap*dz(vz)) = 0'))
    problem.add_equation(('dt(w) + U*dx(w) + V*dy(w) + f*v*sin(tht) + dz(p)'
            '- b*cos(tht) - Pr*(kap*dx(dx(w)) - kap*dy(dy(w)) - dz(kap)*wz'
            '+ kap*dz(wz)) = 0'))
#    problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*N**2*sin(tht)'
#            '+ w*(N**2*cos(tht) + Bz) - kap*dx(dx(b)) - kap*dy(dy(b)) - dz(kap)*bz'
#            '- kap*dz(bz) = 0'))
    problem.add_equation(('dt(b) + U*dx(b) + V*dy(b) + u*Vz*f'
            '+ w*(Bz) - kap*dx(dx(b)) - kap*dy(dy(b)) - dz(kap)*bz'
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
    problem.add_bc('right(w) = 0')
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
    
    # get max growth rates
    gr = np.array([max_growth_rate(0, l) for l in ll])
    
    # get full eigenvectors and eigenvalues for l with largest growth
    idx = sorted_eigen(0., ll[np.argmax(gr)])
    solver.set_state(idx[-1])
    
    # collect eigenvector
    u = solver.state['u']
    v = solver.state['v']
    w = solver.state['w']
    b = solver.state['b']
    
    grt[i] = gr
#%%


#%%
plt.figure(figsize=(4.8, 4.8))
for i in range(size(thtarr)):
    plt.plot(ll*V['g'][-1]/f, grt[i]/f)
    
plt.xlabel('along-track wavenumber [m$^{-1}$]')
plt.ylabel('growth rate')
axes = plt.gca()
axes.set_ylim([0, .3])
axes.set_xlim([0.05, 7])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.tight_layout()
plt.savefig('fig/growth_rate.pdf')
