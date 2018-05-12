#!/usr/bin/env python

# Simple model of receptors diffusing in and out of synapses.
# Simulation of the Dynamcis with the Euler method.
#
# Jochen Triesch, April 2017 - Jaunary 2018

import numpy as np
from  matplotlib import pyplot as plt

# parameters
N = 5               # number of initial conditions to simulate
steps = 10000       # number of time steps to simulate
duration = 60.0     # duration of simulation in minutes
ts = duration/steps # time step of the simulation
beta = 60.0/43.0    # transition rate out of slots in 1/min
delta = 1.0/14.0    # removal rate in 1/min
phi = 2.67          # relative pool size
F_ss = 0.9          # steady state filling fraction
S = 10000.0         # total number of slots

# derived quantities
gamma = delta*F_ss*S*phi
alpha = beta/(phi*S*(1-F_ss))
rho = beta/alpha

# initatlizations
times = np.zeros(steps)
R = np.zeros([N,steps])
for i in range(0, N):
    R[i,0] = 10000*i

# simulation loop
for t in range(0, steps-1):
    F = (0.5*(S+R[:,t]+rho) - np.sqrt(0.25*(S+R[:,t]+rho)**2 - R[:,t]*S))/S
    R[:,t+1] = R[:,t] + ts*delta *(gamma/delta + F*S - R[:,t])
    times[t+1] = ts*(t+1)

# show results
print('Theoretical total number of receptors in steady state:', (1+phi)*F_ss*S)

f = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.plot(times, np.transpose(R),'k')
plt.plot(times[1:round(steps/4)], gamma*times[1:round(steps/4)], 'k:')
plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
plt.ylabel(r'$R$', fontsize=12)
plt.show()
#f.savefig("Fig7.pdf", bbox_inches='tight')
