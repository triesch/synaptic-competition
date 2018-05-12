#!/usr/bin/env python

# Simple model of receptors diffusing in and out of synapses.
# Simulation of the dynamcis with the Euler method.
# Simluates processes occuring during LTP.
# This codes was used to create Fig. 6 of the corresponding paper:
#
# <put reference here>
#
# Jochen Triesch, January 2017 - January 2018

import numpy as np
from  matplotlib import pyplot as plt
import matplotlib.lines as mlines

# parameters
N = 4               # number of synapses
steps = 10000       # number of time steps to simulate
duration = 25.0     # duration in minutes
ts = duration/steps # time step of the simulation

beta = 60.0/43.0    # transition rate out of slots in 1/min
delta = 1.0/14.0    # removal rate in 1/min

T_start = 2.0       # time when LTP induction starts in minutes
T_stimulation = 1.5 # duration of LTP induction protocol (e.g. 45 pulses at 0.5 Hz)

# parameters for alpha time course during LTP
alpha_factor = 4.0      # factor by which alpha of potentiated synapse grows
T_alpha_up = 6.0/60.0   # time during which alpha increases in minutes
T_alpha_down = 2.0      # time during which alpha decreases in minutes

# parameters for gamma time course during LTP
gamma_factor = 1.0      # factor by which exocytosis rate is increased during LTP induction
T_gamma_up = 15.0/60.0  # time during which gamma increases in minutes
T_gamma_down = 1.0      # time during which gamma decreases in minutes

# parameters for slot time course during LTP
S_base = 20.0       # number of slots in smallest synapse
S_inc = 20.0        # increment of slot number between synapses
S_factor = 4.0      # peak factor by which volume of potentiated synapse is inreased (slot number will only increase by the 2/3 power of that)
S_sustained = 2.0   # sustained factor by which volume of potentiated synapse is inreased
a_slot = 8.0        # steepness of sigmoidal volume increase at midpoint of growth.
                    # More precisely, the slope will be given by a/4 (peak-base).
tau_slot = 5.0      # time constant of volume decay from peak to sustained level

# global variables
times = np.zeros(steps)     # absolute times associated with each time step
p = np.zeros(steps)         # pool size
w = np.zeros([N,steps])     # weights
s = np.zeros([N,steps])     # slot numbers
S = 0.0                     # total number of slots
P = 0.0                     # initial pool size
phi = 0.0                   # relative pool size
alpha = np.zeros([N,steps])    
gamma = np.zeros(steps)

def simulate():
    "runs a simulation of the discretized system with the Euler method"

    global S
    
    p[0] = P
    for i in range(0,N):
        w[i,0] = F*s[i,0]
    W = F*S
    print('begin simulation:')
    #print("S =", S, ", W =", W, ", p =", p[0])
    #print("w = ", w[:,0])

    # simulation loop
    for t in range(0, steps-1):
        W = sum(w[:,t])
        S = sum(s[:,t])
        w[:,t+1] = w[:,t] + ts * (alpha[:,t]*p[t] * (s[:,t]-w[:,t]) - beta*w[:,t])
        p[t+1] = p[t] + ts * (beta*W - p[t]*np.sum( np.multiply(alpha[:,t],(s[:,t]-w[:,t]))) - delta*p[t] + gamma[t])
        times[t+1] = ts*(t+1)

    #print('end simulation:')
    #print("S =", S, ", W =", W, ", p =", p[t])
    #print("w = ", w[:,t])


def calculateGammaAndAlpha():
    "calculates the necessary alpha and gamma for the desired pool size and fillig fraction"

    gamma = delta*F*S*phi
    alpha = beta/(phi*S*(1-F))
    return(gamma, alpha)


def setInitialSlots():
    "sets the inital number of slots in all synapses and calculates their sum"

    global S
    
    for i in range(0,N):
        s[i,:] = S_base + i*S_inc
    S = np.sum(s[:,0])

   
def setSlotTimeCourse():
    "sets the time course of the number of slots in all synapses"
    
    # set the time course of number of slots in synapse 2.
    # We assueme that the number of slots goes with the 2/3 power of the volume.
    # So we first convert from slot numbers to volume by taking the 3/2 power
    # and convert back in the end.
    S1_start = s[1,0]**1.5
    S2_start = s[2,0]**1.5
    for n in range(0, steps):
        t = n*ts
        if t < T_start:
            s[1,n] = S1_start**(2.0/3.0)
            s[2,n] = S2_start**(2.0/3.0)
        else:
            if t < T_start+T_stimulation:
               s[1,n] = (S1_start + (S_factor*S1_start - S1_start)*(1/(1+np.exp(-a_slot*(t - T_stimulation/2.0 - T_start)))))**(2.0/3.0)
               s[2,n] = (S2_start + (S_factor*S2_start - S2_start)*(1/(1+np.exp(-a_slot*(t - T_stimulation/2.0 - T_start)))))**(2.0/3.0)
            else:    
                s[1,n] = (S_sustained*S1_start + (S_factor-S_sustained)*S1_start*np.exp(-(t-T_stimulation-T_start)/tau_slot))**(2.0/3.0)
                s[2,n] = (S_sustained*S2_start + (S_factor-S_sustained)*S2_start*np.exp(-(t-T_stimulation-T_start)/tau_slot))**(2.0/3.0)


def setAlphaTimeCourse(base, factor, tStart, tUp, tDown):
    "sets the time course of the insertion rate alpha of synapse 2"

    # alpha starts at "base" value, at "tStart" linearly ramps up by "factor"
    # over time  "tUp", then goes back to "base" over time "tDown"
    
    for n in range(0, steps):
        alpha[:,n] = base # first set all to base, then change for synapse 2 (which has index 1)
        t = n*ts
        if t < tStart:
            alpha[1,n] = base
            alpha[2,n] = base
        else:
            if t < tStart+tUp:
               alpha[1,n] = base*(tStart+tUp-t)/tUp + factor*base*(t-tStart)/tUp
               alpha[2,n] = base*(tStart+tUp-t)/tUp + factor*base*(t-tStart)/tUp
            else:
                if t >= tStart+tUp+tDown:
                    alpha[1,n] = base
                    alpha[2,n] = base
                else:
                   alpha[1,n] = factor*base*(tStart+tUp+tDown-t)/tDown + base*(t-tStart-tUp)/tDown
                   alpha[2,n] = factor*base*(tStart+tUp+tDown-t)/tDown + base*(t-tStart-tUp)/tDown


def setGammaTimeCourse(base, factor, tStart, tUp, tStimulation, tDown):
    "sets the time course of the exocytosis rate gamma (injection rate of receptors into the pool)"

    # gamma starts at "base" value, at "tStart" linearly ramps up by "factor" over time "tUp",
    # stays elevated until end of stimulation "tStimulation" and linearly goes back to base
    # over time "tDown".
    
    for n in range(0, steps):
        t = n*ts
        if t < tStart:
            gamma[n] = base
        else:
            if t < tStart+tUp:
               gamma[n] = base*(tStart+tUp-t)/tUp + base*factor*(t-tStart)/tUp
            else:
                if t < tStart+tStimulation:
                    gamma[n] = base*factor
                else:
                    if t >= tStart+tStimulation+tDown:
                        gamma[n] = base
                    else:
                       gamma[n] = base*(t-tStart-tStimulation)/tDown + base*factor*(tStart+tStimulation+tDown-t)/tDown


def simulateAndDisplaySingleSetting():
    "runs a simulation with a single parameter setting and displays the results"

    global S
    global P
    
    setInitialSlots()
    P = phi*F*S
    
    gamma_base, alpha_base = calculateGammaAndAlpha()
    setSlotTimeCourse()
    setAlphaTimeCourse(alpha_base, alpha_factor, T_start, T_alpha_up, T_alpha_down)
    setGammaTimeCourse(gamma_base, gamma_factor, T_start, T_gamma_up, T_stimulation, T_gamma_down)
    simulate()

    w_rel_homo = (w[1,:]-w[1,0])/w[1,0]*100.0
    w_rel_hetero = (w[0,:]-w[0,0])/w[0,0]*100.0
    homoMax = np.amax(w_rel_homo)
    heteroMax = np.amin(w_rel_hetero)
    print("maximum homosynaptic change: ", homoMax)
    print("maximum heterosynaptic change: ", heteroMax)

    f1 = plt.figure(figsize=(4,3))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)
    plt.rc('font', serif='Times New Roman') 
    [line1, line2, line3, line4] = plt.plot(times, np.transpose(w))
    plt.legend((line4, line3, line2, line1), (r'$w_4$', r'$w_3$', r'$w_2$', r'$w_1$'), loc=1, fontsize=12)
    plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
    plt.ylabel(r'$w_i$', fontsize=12)
    plt.title('$F={0} \, , \; \phi={1}$'.format(F,phi), fontsize=12)
    plt.show()
#    f1.savefig("Fig6B.pdf", bbox_inches='tight')

    f2 = plt.figure(figsize=(4,3))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)
    plt.rc('font', serif='Times New Roman') 
    w_rel_homo = (w[1,:]-w[1,0])/w[1,0]*100.0
    w_rel_hetero = (w[0,:]-w[0,0])/w[0,0]*100.0
    p_rel = (p[:]-p[0])/p[0]*100.0
    plt.plot(times, w_rel_homo,'g', label=r'homosynaptic')
    plt.plot(times, w_rel_hetero,'b', label=r'heterosynaptic')
    plt.plot(times, p_rel, 'k', label=r'pool size')
    plt.axis((0.0, 25.0, -80.0, 180.0))
    plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
    plt.ylabel(r'relative change (%)', fontsize=12)
    plt.title('$F={0} \, , \; \phi={1}$'.format(F,phi), fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.show()
#    f2.savefig("Fig6D.pdf", bbox_inches='tight')

    f3, ax1 = plt.subplots(figsize=(4,3))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)
    plt.rc('font', serif='Times New Roman')
    plt.plot(times, alpha[1,:]/alpha_base*100.0,'k', label=r'$\alpha$')
    plt.plot(times, s[1,:]/s[1,0]*100.0,'r', label=r'slots')
    plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
    plt.ylabel(r'homosynaptic change (%)', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.show()
#    f3.savefig("Fig6A.pdf", bbox_inches='tight')


def simulateVaryingPhi():
    "runs multiple simulations for varying phi (and F) and displays the results"

    global F
    global P
    global S
    global phi
    
    F_steps = 3
    phi_steps = 49
    homoMax = np.zeros([F_steps, phi_steps])
    heteroMax = np.zeros([F_steps, phi_steps])
    F_values = np.zeros(F_steps)
    phi_values = np.zeros(phi_steps)
    for f in range(0, F_steps):
        F = 0.5 + 0.2*f
        F_values[f] = F
        for p in range(0, phi_steps):
            phi = 0.1 + 0.1*p
            phi_values[p] = phi
            setInitialSlots()
            P = phi*F*S
            gamma_base, alpha_base = calculateGammaAndAlpha()
            setSlotTimeCourse()
            setAlphaTimeCourse(alpha_base, alpha_factor, T_start, T_alpha_up, T_alpha_down)
            setGammaTimeCourse(gamma_base, gamma_factor, T_start, T_gamma_up, T_stimulation, T_gamma_down)
            print('F = ', F, ', phi = ', phi)
            simulate()
            w_rel_homo = (w[1,:]-w[1,0])/w[1,0]*100.0
            w_rel_hetero = (w[0,:]-w[0,0])/w[0,0]*100.0
            homoMax[f,p] = np.amax(w_rel_homo)
            heteroMax[f,p] = np.amin(w_rel_hetero)
            print("maximum homosynaptic change: ", homoMax[f,p])
            print("maximum heterosynaptic change: ", heteroMax[f,p])

    # plot results
    f1 = plt.figure(figsize=(4,3))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)
    plt.rc('font', serif='Times New Roman') 
    plt.plot(phi_values, homoMax[0,:],'b')
    plt.plot(phi_values, homoMax[1,:],'k')    
    plt.plot(phi_values, homoMax[2,:],'r')    
    plt.legend([r'$F=0.5$', r'$F=0.7$', r'$F=0.9$'], loc='lower right', fontsize=12)
    plt.xlabel(r'$\phi$', fontsize=12)
    plt.ylabel(r'max. homosynpatic change (%)', fontsize=12)
    plt.show()
#    f1.savefig("Fig6E.pdf", bbox_inches='tight')

    f2 = plt.figure(figsize=(4,3))
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)
    plt.rc('font', serif='Times New Roman') 
    plt.plot(phi_values, heteroMax[0,:],'b')
    plt.plot(phi_values, heteroMax[1,:],'k')    
    plt.plot(phi_values, heteroMax[2,:],'r')
    plt.legend([r'$F=0.5$', r'$F=0.7$', r'$F=0.9$'], loc='lower right', fontsize=12)
    plt.xlabel(r'$\phi$', fontsize=12)
    plt.ylabel(r'max. heterosynpatic change (%)', fontsize=12)
    plt.show()
#    f2.savefig("Fig6F.pdf", bbox_inches='tight')


# Main program

# to simulate a single parameter setting use:
#F = 0.9         # default value 0.9
#phi = 2.67      # default value 2.67
#simulateAndDisplaySingleSetting()

# to create the plots for varying phi and F use:
simulateVaryingPhi()

