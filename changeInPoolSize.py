#!/usr/bin/env python

# Simple model of receptors diffusing in and out of synapses.
# Simulation of the Dynamcis with the Euler method.
# This simulates the effect of a sudden change in the pool size
#
# Jochen Triesch, January-April 2017

import numpy as np
from matplotlib import pyplot as plt

# parameters
N = 3               # number of synapses
steps = 10000       # number of time steps to simulate
duration = 10.0     # duration in minutes
change_time = 2.0   # time at which number of pool size changes in minutes
ts = duration/steps # time step of the simulation
beta = 60.0/43.0    # transition rate out of slots in 1/min
delta = 1.0/14.0    # removal rate in 1/min
phi = 2.67          # relative pool size
F = 0.9             # set desired filling fraction

# initializations: the w_i and p are set to their steady state values
s = np.zeros(N)
for i in range(0,N):
    s[i] = 40.0 + i*20.0
S = sum(s)

gamma = delta*F*S*phi       # production rate set to achieve desired p*
alpha = beta/(phi*S*(1-F))  # set alpha accordingly
P = gamma/delta             # total number of receptors in steady state

# variables we want to keep track of to plot them at the end:
# 'u' stands for up-regulation and 'd' stands for down-regulation.
# Up- and down-regulation are simulated simultaneously.
pu = np.zeros(steps)        # pool size
pd = np.zeros(steps)
wu = np.zeros([N,steps])    # synaptic weights
wd = np.zeros([N,steps])
ru = np.zeros(steps)        # relative change of synaptic weights
rd = np.zeros(steps)
times = np.zeros(steps)

pu[0] = P
pd[0] = P
ru[0] = 1.0
rd[0] = 1.0

for i in range(0,N):
    wu[i,0] = F*s[i]
    wd[i,0] = F*s[i]


# simulation loop
for t in range(0, steps-1):

    if t==round(change_time/ts): # change pool size after some time
       pu[t]=2.0*P # double number of receptors in the pool
       pd[t]=0.0*P # halve number of receptors in the pool

    Wu = sum(wu[:,t])
    Wd = sum(wd[:,t])
    wu[:,t+1] = wu[:,t] + ts * (alpha*pu[t] * (s-wu[:,t]) - beta*wu[:,t])
    wd[:,t+1] = wd[:,t] + ts * (alpha*pd[t] * (s-wd[:,t]) - beta*wd[:,t])
    pu[t+1] = pu[t] + ts * (beta*Wu - alpha*pu[t]*(S-Wu) - delta*pu[t] + gamma)
    pd[t+1] = pd[t] + ts * (beta*Wd - alpha*pd[t]*(S-Wd) - delta*pd[t] + gamma)
    ru[t+1] = wu[0,t+1]/wu[0,0]*100.0
    rd[t+1] = wd[0,t+1]/wd[0,0]*100.0
    times[t+1] = ts*(t+1)

# show results
f = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.gca().set_prop_cycle(plt.cycler('color', ['blue', 'green', 'red']))
[line1, line2, line3] = plt.plot(times, np.transpose(wu))
plt.plot(times, np.transpose(wd), ls='dotted')
plt.legend((line3, line2, line1), (r'$w_3$', r'$w_2$', r'$w_1$'), loc=1, fontsize=12)
plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
plt.ylabel(r'$w_i$', fontsize=12)
plt.title(r'$F=0.9$', fontsize=12)
plt.show()
f.savefig("Fig4A.pdf", bbox_inches='tight')

f2 = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman')
plt.plot(times, pu, "k")
plt.plot(times, pd, "k", ls='dotted')
plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
plt.ylabel('pool size', fontsize=12)
plt.title(r'$F=0.9$', fontsize=12)
plt.show()
f2.savefig("Fig4C.pdf", bbox_inches='tight')

f3 = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.plot(times, ru, "k")
plt.plot(times, rd, "k", ls='dotted')
plt.axis((0.0, 10.0, 40.0, 140.0))
plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
plt.ylabel(r'$w_i(t)/w_i(0) \quad [\%]$', fontsize=12)
plt.title(r'$F=0.9$', fontsize=12)
plt.show()
f3.savefig("Fig4B.pdf", bbox_inches='tight')
