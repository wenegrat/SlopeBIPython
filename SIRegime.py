import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
import scipy.integrate as integrate

import logging
logger = logging.getLogger(__name__)

# parameters
N = 1e-4 # buoyancy frequency
Pr = 0.1
f = Pr*N # Coriolis parameter
tht = -5e-2 # slope angle
H = 100. # domain height
Ri =1.25 # CI PARAMETERS
S = 2.75

Ri = 0.5  #SI PARAMETERS
S = 0.1
Lmd = N/Ri**(1/2) # shear


tht = np.arctan(f/N*S**(1/2))
# slope parameter
print(np.tan(tht)*N**2/(f*Lmd))

# along-slope wavenumbers
ll = np.linspace(.01*f/(N*H), 20*f/(N*H), 2)

# number of grid points
nz = 128

# file name that results are saved in
name = 'test'

# build domain
z_basis = de.Chebyshev('z', nz, interval=(0, H))
domain = de.Domain([z_basis], np.complex128)

# LINEAR STABILITY

problem = de.EVP(domain, variables=['u', 'v', 'w', 'b', 'p'], eigenvalue='omg')
problem.parameters['N'] = N
problem.parameters['f'] = f
problem.parameters['tht'] = tht
problem.parameters['Lmd'] = Lmd
problem.parameters['k'] = 0. # will be set in loop
problem.parameters['l'] = 0. # will be set in loop
problem.substitutions['dx(A)'] = "1j*k*A"
problem.substitutions['dy(A)'] = "1j*l*A"
problem.substitutions['dt(A)'] = "-1j*omg*A"
problem.add_equation('dt(u) + Lmd*z/cos(tht)*dy(u) - f*v*cos(tht) + dx(p) - b*sin(tht) +dx(p) = 0')
problem.add_equation('dt(v) + Lmd*z/cos(tht)*dy(v) + w*Lmd/cos(tht) + f*u*cos(tht) - f*w*sin(tht) + dy(p) = 0')
problem.add_equation('(dt(w) + Lmd*z/cos(tht)*dy(w)) + f*v*sin(tht) + dz(p) - b*cos(tht) = 0')
problem.add_equation('dt(b) + Lmd*z/cos(tht)*dy(b) + u*(N**2*sin(tht) + f*Lmd*cos(tht)) + w*(N**2*cos(tht) - f*Lmd*sin(tht)) = 0')
problem.add_equation('dx(u) + dy(v) + dz(w) = 0')
problem.add_bc('left(w) = 0')
problem.add_bc('right(w) = 0')

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
gr = np.array([max_growth_rate(l, 0) for l in ll])

# get full eigenvectors and eigenvalues for l with largest growth
idx = sorted_eigen(ll[np.argmax(gr)], 0.)
solver.set_state(idx[-1])

# collect eigenvector
u = solver.state['u']
v = solver.state['v']
w = solver.state['w']
b = solver.state['b']

z = domain.grid(0)


# PLOTTING

# mean state
#%%
plt.figure(figsize=(4.8, 4.8))
plt.plot(ll*abs(Lmd*H/f), gr/abs(f))
plt.xlabel('along-slope wavenumber')
plt.ylabel('growth rate')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.tight_layout()
#plt.savefig('fig/growth_rate.pdf')

# most unstable mode
#%%

# Background buoyancy field
k = ll[np.argmax(gr)]
xr = np.linspace(0, 2*pi/k, nz)
B = np.zeros((nz, nz))
counter = 0
for x in xr:
    B[:, counter] = (N**2*np.sin(tht) + f*Lmd*np.cos(tht))*x + z*(N**2*np.cos(tht) - f*Lmd*np.sin(tht))
    counter = counter+ 1

ly = np.linspace(0, 2*np.pi, nz)

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6.4, 6.4))
im = ax[0,0].pcolormesh(ly, z/H, np.real(u['g'].reshape(nz, 1) * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[0,0])
ax[0,0].set_title('across-slope velocity')
im = ax[0,1].pcolormesh(ly, z/H, np.real(v['g'].reshape(nz, 1) * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')

plt.colorbar(im, ax=ax[0,1])
ax[0,1].set_title('along-slope velocity')
im = ax[1,0].pcolormesh(ly, z/H, np.real(w['g'].reshape(nz, 1) * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[1,0])
ax[1,0].set_title('slope-normal velocity')
im = ax[1,1].pcolormesh(ly, z/H, np.real(b['g'].reshape(nz, 1) * np.exp(1j*ly.reshape(1,nz))), rasterized=True, cmap='RdBu_r')
plt.colorbar(im, ax=ax[1,1])
ax[1,1].set_title('buoyancy')
ax[0,0].set_xticks([0, np.pi, 2*np.pi])
ax[1,0].set_xlabel('phase')
ax[1,1].set_xlabel('phase')
ax[0,0].set_ylabel('slope-normal coordinate')
ax[1,0].set_ylabel('slope-normal coordinate')

ax[1, 0].contour(ly, z/H, B, colors='k')
ax[0, 1].contour(ly, z/H, B, colors='k')
ax[0, 0].contour(ly, z/H, B, colors='k')
ax[1, 1].contour(ly, z/H, B, colors='k')
#plt.savefig('fig/modes.pdf', dpi=300)

plt.show()

#%%
# shear production

# In cartesian frame
Vzhat = Lmd
VSP = -2*np.real((w['g'])*np.conj(v['g'])*Vzhat*np.cos(tht) + np.conj(v['g'])*u['g']*Vzhat*np.sin(tht))
Vx = -Lmd/np.cos(tht)*np.sin(tht)
LSP = -2*np.real(np.conj(v['g'])*u['g']*np.cos(tht) -(w['g'])*np.conj(v['g'])*np.sin(tht))*Vx

# In rotated frame
#Vx = -Lmd/np.cos(tht)*np.sin(tht)
#Vzhat = Lmd*np.cos(tht) - Vx*np.sin(tht)
#VSP = -2*np.real(np.conj(w['g'])*v['g']*Vzhat )
#LSP =  -2*np.real(np.conj(v['g'])*u['g'])*Vx


# buoyancy production
BP = 2*np.real((u['g']*np.sin(tht)+w['g']*np.cos(tht))*np.conj(b['g']))
HHF = -2*np.real(f*Lmd/N**2*(u['g']*np.cos(tht)-w['g']*np.sin(tht))*np.conj(b['g']))


fs = 16
plt.figure(figsize=(5, 5))
plt.plot(BP, z)
plt.plot((VSP), z)
plt.plot(LSP, z)
plt.plot(HHF-BP, z)
plt.plot(VSP+BP+LSP, z)
plt.xlabel('Kinetic Energy Tendency', fontsize=fs)
plt.ylabel('slope-normal coordinate [m]', fontsize=fs)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.legend(['Buoyancy Production', 'VSP', 'LSP', '$PE_t$', '$KE_t$'], frameon=False, fontsize=fs, loc=1)
plt.tight_layout()
plt.ylim((0, H))
plt.grid(linestyle='--', alpha = 0.5)

print('Ri: ' + str(Ri) + '  S: ' + str(S))
LV = N**2/(f*Lmd)*np.tan(tht)*(1- (f/N)**2*((1-Lmd/f*np.tan(tht))))
LSPVSPt = LV
print('LSP/VSP Theory: ' +str(LSPVSPt))
phit = 1/2*np.arctan(   2*f*Lmd/(N**2- f**2-f*Vx)) +np.pi/2*0
LSPVSPthomas = -f*Vx/(f*Lmd)*1/np.tan(phit)
print('LSP/VSP Thom.  : ' + str(LSPVSPthomas))
alpha = -Vx/f
beta = Lmd/f
phidewar = f*Lmd/N**2*(1+f**2/N**2*(1-alpha)) - np.pi/2*0
LDewar = (beta/alpha*np.tan(phidewar))**(-1)
VSPDewar = np.sin(phidewar)*(beta*np.sin(phidewar)-np.cos(phidewar)*(1-alpha))*beta
LSPDewar = np.cos(phidewar)*(beta*np.sin(phidewar)-np.cos(phidewar)*(1-alpha))*alpha
print('LSP/VSP Dewar: ' +str(LDewar))

LSPVSP = integrate.trapz(LSP, domain.grid(0))/integrate.trapz(VSP, domain.grid(0))
print('LSP/VSP Num.  : ' + str(LSPVSP))


vort = f+Vx
print('Vert Vort: '+ str(vort))
pv = (f+Vx)*N**2 - Lmd*f*Lmd
print('PV : ' + str(pv))
phi = np.arctan(-1/Ri)*180/np.pi
print('Phi :' + str(phi))

#%%
# Background buoyancy field
#k = ll[np.argmax(gr)]
#xr = np.linspace(0, 2*np.pi/k, nz)
#VSPB = np.zeros((nz, nz))+1j*np.zeros((nz,nz))
#LSPB = np.zeros((nz, nz))+1j*np.zeros((nz,nz))
#
#counter = 0
#for x in xr:
#    wt = np.real(w['g']*np.exp(1j*k*x))
#    vt = np.real(v['g']*np.exp(1j*k*x))
#    ut = np.real(u['g']*np.exp(1j*k*x))
#    bt = np.real(b['g']*np.exp(1j*k*x))
#
#    VSPB[:, counter] = -2*(wt*vt*Vzhat*np.cos(tht) + vt*ut*Vzhat*np.sin(tht))
#    LSPB[:, counter] = -2*(vt*ut*np.cos(tht) -wt*vt*np.sin(tht))*Vx
#    counter = counter+ 1
#
#VSPm = integrate.trapz((VSPB), xr, axis=1)
#LSPm = integrate.trapz((LSPB), xr, axis=1)
#
#plt.figure()
#plt.plot(VSPm, z)
#plt.plot(LSPm, z)
#plt.plot(LSP, z)
#
#print(str(integrate.trapz(LSPm, domain.grid(0))/integrate.trapz(VSPm, domain.grid(0))))



#%% Define stream function
ly = np.linspace(0, 2*np.pi, nz)
wf = np.real(u['g'].reshape(nz, 1) * np.exp(1j*ly.reshape(1,nz)))
phi = np.zeros((nz, nz))
phi[0:-1,:] = integrate.cumtrapz(wf, z, axis=0)
#ly = np.linspace(0, 1, nz)
xs = np.linspace(0, 2*np.pi/k, nz)
xr = np.linspace(0, 2*pi/k, nz)
B = np.zeros((nz, nz))
counter = 0
for x in xr:
    B[:, counter] = (N**2*np.sin(tht) + f*Lmd*np.cos(tht))*x + z*(N**2*np.cos(tht) - f*Lmd*np.sin(tht))
    counter = counter+ 1
    
fig, ax = plt.subplots(1,2, figsize=(12, 7.5),sharey=True, gridspec_kw = {'width_ratios':[2,1]})
#plt.rcParams['contour.negative_linestyle'] = 'solid'
#ax[0].contour(ly, z/H, phi,4, colors='k')
ax[0].contourf(ly, z/H, phi, 20,cmap='RdBu_r')
ax[0].contour(ly, z/H, B, colors='k')
ax[0].set_xlabel('Across-slope phase')
ax[0].set_ylabel('z/H')
#ax[0].set_aspect('equal', 'datalim')
#ax[0].set_xlim((0, 1))
ax[1].plot(BP, z/H)
ax[1].plot((VSP), z/H)
ax[1].plot(LSP, z/H)
#ax[1].plot(HHF-BP, z)
#ax[1].plot(VSP+BP+LSP, z)
ax[1].set_xlabel('Kinetic Energy Tendency', fontsize=fs)
#ax[1].set_ylabel('slope-normal coordinate [m]', fontsize=fs)
ax[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax[1].legend(['BP', 'VSP', 'LSP', '$PE_t$', '$KE_t$'], frameon=False, fontsize=fs, loc=1)
ax[1].set_ylim((0, 1))
ax[1].grid(linestyle='--', alpha = 0.5)
#plt.tight_layout()

ax[0].set_xticks([0, np.pi, 2*np.pi])

labels = ['0', '$\pi$', '$2\pi$']
ax[0].set_xticklabels(labels)  

plt.savefig('/home/jacob/Dropbox/Slope BI/Slope BI Manuscript/SIExample.eps', format='eps', dpi=1000)


