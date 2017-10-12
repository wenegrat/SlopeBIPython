#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 14:46:14 2017

@author: jacob
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
from dedalus import public as de

# Input
data = './snapshots/snapshots_s1.h5' #e.g. snapshots/snapshots_s1.h5
ntosave = 10.

# Bases and domain
Lx, Ly, Lz = (1e4, 1e4, 100)
DEG2DIST = np.pi * 6.37e6 / 180
x_basis = de.Fourier('x', 32, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', 32, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', 16, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64)
x = domain.grid(0) 
y = domain.grid(1) 
z = domain.grid(2)
Y_yz, Z_yz = np.meshgrid(y,z,indexing='ij')
X_xy, Y_xy = np.meshgrid(x,y,indexing='ij')

# Figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
f= h5py.File(data, mode='r')

u = f['tasks']['u midplane']
w = f['tasks']['w midplane']
bs = f['tasks']['b surface']
b = f['tasks']['b midplane']
#%%
pcolor(bs[-1,:,:,0])
#pcolor(w[8,:,0,:])
colorbar
#pcolor(f['tasks']['b midplane'][-1, :,0, :])
#%%
# Make plot with vertical (default) colorbar
fig, ax = plt.subplots()

data = b[6,:,0,:]

cax = ax.imshow(data, interpolation='nearest')
ax.set_title('w')

# Add colorbar, make sure to specify tick locations to match desired ticklabels
cbar = fig.colorbar(cax)