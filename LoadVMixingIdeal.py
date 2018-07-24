#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:05:03 2018

@author: jacob
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# LOAD
a = np.load('/home/jacob/dedalus/MixingStabilityIdeal/StabilityData_0.002.npz');

gr = a['gr']
gr[gr>1e-5] = 0
ll = a['ll']

plt.figure()
plt.semilogx(ll/(2*np.pi), gr)

ind = np.argmax(gr)
idealwave = 2*np.pi/ll[ind]
idealgr = 1/gr[ind]/86400