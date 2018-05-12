#!/usr/bin/env python

# Simple model of receptors diffusing in and out of synapses.
# Simulation of the Dynamcis with the Euler method.
#
# Jochen Triesch, January-July 2017

import numpy as np
from  matplotlib import pyplot as plt
import matplotlib.lines as mlines

# parameters
N = 4               # number of synapses
steps = 10000       # number of time steps to simulate
duration = 10.0     # duration in minutes
change_time = 2.0   # time at which number of slots changes
ts = duration/steps # time step of the simulation
beta = 60.0/43.0    # transition rate out of slots in 1/min
delta = 1.0/14.0    # removal rate in 1/min
F = 0.5             # desired filling fraction
phi = 2.67          # relative pool size

# variables
p = np.zeros(steps)
w = np.zeros([N,steps])
s = np.zeros([N,steps])
times = np.zeros(steps)

# set initial numbers of slots and receptors, weights, pool size
for i in range(0,N):
    s[i,0] = 20.0 + i*20.0
S = np.sum(s[:,0])
W = F*S             # initial sum of weights
P = phi*W           # receptors in the pool in steady state
p[0] = P
for i in range(0,N):
    w[i,0] = F*s[i,0]

# derived parameters
gamma = delta*F*S*phi       # production rate set to achieve desired pool size
alpha = beta/(phi*S*(1-F))  # set alpha accordingly


print('beginning:')
print("S =", S, ", W =", W, ", p =", p[0])
print("w = ", w[:,0])

# simulation loop
for t in range(0, steps-1):

    if t==round(change_time/ts): # alter number of slots after some time
      s[N-2,t] = s[N-2,t]*2.0
      s[N-4,t] = s[N-4,t]*2.0
        
    W = sum(w[:,t])
    S = sum(s[:,t])
    s[:,t+1] = s[:,t]
    w[:,t+1] = w[:,t] + ts * (alpha*p[t] * (s[:,t]-w[:,t]) - beta*w[:,t])
    p[t+1] = p[t] + ts * (beta*W - alpha*p[t]*(S-W) - delta*p[t] + gamma)
    times[t+1] = ts*(t+1)


# show results
print('end:')
print("S =", S, ", W =", W, ", p =", p[t])
print("w = ", w[:,t])

f1 = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
[line1, line2, line3, line4] = plt.plot(times, np.transpose(w))
plt.axis((0.0, 10.0, 0.0, 60.0))
plt.title(r'$F=0.5$', fontsize=12)
plt.legend((line4, line3, line2, line1), (r'$w_4$', r'$w_3$', r'$w_2$', r'$w_1$'), loc=1, fontsize=12)
plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
plt.ylabel(r'$w_i$', fontsize=12)
plt.show()
#f1.savefig("Fig5A.pdf", bbox_inches='tight')

f2 = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
w_rel_homo = (w[0,:]-w[0,0])/w[0,0]*100.0
w_rel_hetero = (w[1,:]-w[1,0])/w[1,0]*100.0
p_rel = (p[:]-p[0])/p[0]*100.0
plt.plot(times, w_rel_homo,'b', label=r'homosynaptic')
plt.plot(times, w_rel_hetero,'g', label=r'heterosynaptic')
plt.plot(times, p_rel, 'k', label=r'pool size')
plt.title(r'$F=0.5$', fontsize=12)
plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
plt.ylabel(r'relative change (%)', fontsize=12)
plt.legend(loc='upper right', fontsize=12)
plt.show()
#f2.savefig("Fig5B.pdf", bbox_inches='tight')


