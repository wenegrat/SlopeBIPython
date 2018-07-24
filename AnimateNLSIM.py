#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AnimateLatmix.py
Created on Mon Mar 26 09:43:29 2018

@author: jacob
"""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.pyplot.rcParams['animation.mencoder_path'] = '/usr/bin/mencoder'

import matplotlib.animation as animation
import matplotlib.ticker 

import h5py
import scipy.io as spio
import scipy.interpolate as interpolate
from cmocean import cm as cmo

#%% LOAD NL SIM SNAPS
filename = '/home/jacob/dedalus/NLSIM/snapshots/snapshots_nlfast1.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

# Get the data
data = list(f['tasks'])

b = f['tasks']['b']


x = b.dims[1][0][:]
y = b.dims[2][0][:]
z = b.dims[3][0][:]
time = b.dims[0][0][:]
nt = time.size
tht = 1e-2

#%% Interpolate to uniform time steps
nt, nx, ny, nz = b.shape
timei = np.arange(0, time[-1], 1800)
bi = np.zeros((timei.size, nx, ny))
for i in range(0, x.size):
    for j in range(0, y.size):
        #interpb
        bt = interpolate.interp1d(time, b[:,i,j,36])
        bi[:,i,j] = bt(timei)
        
#%% MAKE MOVIE
nc = 500 #75
cm = cmo.ice

#mpl.rcParams['text.usetex'] = True
mpl.rcParams.update({'font.size': 32})
#mpl.rcParams['mathtext.fontset'] = 'stix'
#mpl.rcParams['font.family'] = 'STIXGeneral'


var = bi
time = timei
bcl = np.linspace(-1e-4, 3.7e-3, nc)

bc = np.linspace(-1e-3, 3.5e-3, 10)
fig = mpl.pyplot.figure(figsize=(10, 10))
ax1 = mpl.pyplot.axes()

#Writer = mpl.animation.writers['mencoder']
#writer = Writer(fps=12, metadata=dict(artist='Me'),extra_args=["-really-quiet"], bitrate=4000)
#writer.dpi=1080
Writer = mpl.animation.writers['ffmpeg']
writer = Writer(fps=36, metadata=dict(artist='Me'), bitrate=10000)


#We need to prime the pump, so to speak and create a quadmesh for plt to work with
#quad1 = ax1.pcolormesh(x, y, np.transpose(var[0,:,:])+(3.4e-3)**2*np.sin(tht)*x, vmin=bcl[0], vmax=bcl[-1], cmap=cm)
quad1 = ax1.contourf(x/1000, y/1000, np.transpose(var[0,:,:])+(3.4e-3)**2*np.sin(tht)*x, bcl, cmap=cm, extend='both')
#quad2 = ax1.contour(x, y, np.transpose(var[0,:,:])+(3.4e-3)**2*np.sin(tht)*x, bc, color=1)
ax1.set_title('Buoyancy,$\;\;\;\;\;$ day: %1.1f' %(time[0]/86400))
ax1.set_aspect('equal')
f = lambda x,pos: str(np.ceil(x)).rstrip('0').rstrip('.')

ax1.set_xticks((np.linspace(x[0], x[-1], 3)/1000))
ax1.set_yticks((np.linspace(y[0], y[-1], 3)/1000))
ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xlabel('Across-slope, km')
ax1.set_ylabel('Along-slope, km')
def animate(i):
    global quad1
#    global quad2
#    tn = time[i]/86400
#    mpl.pyplot.title('Time: %.2f'%tn)
    print(i)
#    dyn = np.transpose(var[i,:,:])+(3.4e-3)**2*np.sin(tht)*x#/np.max(var[i,:,:])
    #This is where new data is inserted into the plot.
#    quad1.set_array(dyn.ravel())
#    quad2.set_array(b[i,:,:].ravel())
    for c in quad1.collections:
        c.remove()
    quad1 = ax1.contourf(x/1000, y/1000, np.transpose(var[i,:,:])+(3.4e-3)**2*np.sin(tht)*x, bcl, cmap=cm, extend='both')
    ax1.set_title('Buoyancy,$\;\;\;\;\;$ day: %1.1f' %(time[i]/86400))
#    for c in quad2.collections:
#        c.remove()
#    quad2 = ax1.contour(x, y, np.transpose(var[i,:,:])+(3.4e-3)**2*np.sin(tht)*x, bc, colors='1.0')
    
    return  quad1

# Frames argument is how many time steps to call 'animate' function
anim = mpl.animation.FuncAnimation(fig, animate, frames = [x for x in range(0, 1921) if x % 1 == 0], blit = False)

# Turn this line on to save to file
#anim.save('b_anim_fast_interp.mp4', writer=writer) # Uncomment to Save Animation

#mpl.pyplot.show() # Can turn this on to just show the animation on screen
print('finished')


##%% SAVE MOVIE PCOLOR
#nc = 500 #75
#cm = cmo.ice
#
##mpl.rcParams['text.usetex'] = True
#mpl.rcParams.update({'font.size': 32})
##mpl.rcParams['mathtext.fontset'] = 'stix'
##mpl.rcParams['font.family'] = 'STIXGeneral'
#
#
#var = bi
#time = timei
#bcl = np.linspace(-1e-4, 3.7e-3, nc)
#
#bc = np.linspace(-1e-3, 3.5e-3, 10)
#fig = mpl.pyplot.figure(figsize=(10, 10))
#ax1 = mpl.pyplot.axes()
#
##Writer = mpl.animation.writers['mencoder']
##writer = Writer(fps=12, metadata=dict(artist='Me'),extra_args=["-really-quiet"], bitrate=4000)
##writer.dpi=1080
#Writer = mpl.animation.writers['ffmpeg']
#writer = Writer(fps=36, metadata=dict(artist='Me'), bitrate=10000)
#
#
##We need to prime the pump, so to speak and create a quadmesh for plt to work with
#quad1 = ax1.pcolormesh(x/1000, y/1000, (var[0,:,:])+(3.4e-3)**2*np.sin(tht)*x, vmin=bcl[0], vmax=bcl[-1], cmap=cm, shading='gourad')
##quad1 = ax1.contourf(x/1000, y/1000, np.transpose(var[0,:,:])+(3.4e-3)**2*np.sin(tht)*x, bcl, cmap=cm, extend='both')
##quad2 = ax1.contour(x, y, np.transpose(var[0,:,:])+(3.4e-3)**2*np.sin(tht)*x, bc, color=1)
#ax1.set_title('Buoyancy,$\;\;\;\;\;$ day: %1.1f' %(time[0]/86400))
#ax1.set_aspect('equal')
#f = lambda x,pos: str(np.ceil(x)).rstrip('0').rstrip('.')
#
#ax1.set_xticks((np.linspace(x[0], x[-1], 3)/1000))
#ax1.set_yticks((np.linspace(y[0], y[-1], 3)/1000))
#ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
#ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
#ax1.spines['top'].set_visible(False)
#ax1.spines['left'].set_visible(False)
#ax1.spines['right'].set_visible(False)
#ax1.spines['bottom'].set_visible(False)
#ax1.set_xlabel('Across-slope, km')
#ax1.set_ylabel('Along-slope, km')
#def animate(i):
#    global quad1
##    global quad2
##    tn = time[i]/86400
##    mpl.pyplot.title('Time: %.2f'%tn)
#    print(i)
#    dyn = np.transpose((var[i,:,:])+(3.4e-3)**2*np.sin(tht)*x)#/np.max(var[i,:,:])
#    #This is where new data is inserted into the plot.
#    quad1.set_array(dyn.ravel(order='C'))
##    quad2.set_array(b[i,:,:].ravel())
##    for c in quad1.collections:
##        c.remove()
##    quad1 = ax1.contourf(x/1000, y/1000, np.transpose(var[i,:,:])+(3.4e-3)**2*np.sin(tht)*x, bcl, cmap=cm, extend='both')
#    ax1.set_title('Buoyancy,$\;\;\;\;\;$ day: %1.1f' %(time[i]/86400))
##    for c in quad2.collections:
##        c.remove()
##    quad2 = ax1.contour(x, y, np.transpose(var[i,:,:])+(3.4e-3)**2*np.sin(tht)*x, bc, colors='1.0')
#    
#    return  quad1
#
## Frames argument is how many time steps to call 'animate' function
#anim = mpl.animation.FuncAnimation(fig, animate, frames = [x for x in range(0, 1921) if x % 80 == 0], blit = False)
##anim.save('b_anim_fast_interp.mp4', writer=writer) # Uncomment to Save Animation
#
##mpl.pyplot.show() # Can turn this on to just show the animation on screen
#print('finished')
#
#
#
