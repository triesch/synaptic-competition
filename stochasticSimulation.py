# stochastic simulation of AMPA receptor dynamics using the Gillespie algorithm

# This program was used to create Fig. 8A,B and the data for 8C,D of the manuscript.
# It plots w_i of synapses during an example simulation.
# Furthermore it plots the CV  of w_i of 7 synapses. 
# For combining the results from multiple simulations into one figure (like Fig.8C,D) please run coefficientOfVariation.py.

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from copy import deepcopy
import time

###################################################
######## functions ################################
###################################################

def next_values(a0,a,r1,r2):
    """returns values for the next reaction like time difference and reaction according to Gillespie"""
    
    # calculate next time
    new_time_difference = (1/a0)*np.log(1/r1)
    
    # choose next reaction R_mu under the condition that
    # sum(a[i], i=0, mu-1) < r2*a0 <= sum(a[i], i=0, mu)
    mu = 0
    N = r2*a0 - a[mu] 
    
    while N > 0:
        mu += 1
        N = N - a[mu]
    
    return(new_time_difference, mu)
    
def calculate_hi(n,m):
    """calculates the hi with the help of binomial coeff and factorial() 
    since hi is defined as total number of distinct 
    combinations of Ri reactant molecules"""
    
    b=[0]*(n+1)
    b[0]=1
    for i in range(1,n+1):
        b[i]=1
        j=i-1
        while j>0:
            b[j]+=b[j-1]
            j-=1
    hi = b[m]*factorial(m)
    return(hi)
    
def reactions_stoch(reactions):
    """gets a string of several reactions and outputs the stoichiometry array
    of substrates and products
    
    the form of the string is like '2X1+X2->X3,X1->0,X3+4X2->12X1'
    to define N species please use X1, X2, X3,..., XN-1, XN 
    only use symbols like '->' and '+', dont use spaces
    """
    
    # important variables
    substrates = [] # string reactions is splitted into substrates and products
    products = []
    sub_without_plus = [] # each subtrate is separated by '+'
    prod_without_plus = []
    sub_number_reaction = [] # list with items like [number of species, name of species]
    prod_number_reaction = []
    total_number_species = 0 # number of all species to create array
    
    # remove symobls like '+', '->', ',' and split it into substrates and products 
    one_reaction = reactions.split(",")
    for i in one_reaction:
        s_p = i.split("->")
        substrates.append(s_p[0])
        products.append(s_p[1])

    for i in substrates:
        sub_without_plus.append(i.split("+"))
    for i in products:
        prod_without_plus.append(i.split("+"))
    
    # split each item into [number of species, name of species]
    for i in sub_without_plus:
        sub_one_reaction = []
        for j in i:
            number_reaction = []
            j = j.split("X")
            if j[0].isdigit():
                number_reaction.append(int(j[0]))
            else:
                number_reaction.append(1)
            if len(j) == 2:
                number_reaction.append(int(j[1]))
                if int(j[1]) > total_number_species:
                    total_number_species = int(j[1])
            if len(number_reaction) == 2:
                sub_one_reaction.append(number_reaction)
        if len(sub_one_reaction) >= 1:
            sub_number_reaction.append(sub_one_reaction)
    
    for i in prod_without_plus:
        prod_one_reaction = []
        for j in i:
            number_reaction = []
            j = j.split("X")
            if j[0].isdigit():
                number_reaction.append(int(j[0]))
            else:
                number_reaction.append(1)
            if len(j) == 2:
                number_reaction.append(int(j[1]))
                if int(j[1]) > total_number_species:
                    total_number_species = int(j[1])
            if len(number_reaction) == 2:
                prod_one_reaction.append(number_reaction)
        if len(prod_one_reaction) >= 1:
            prod_number_reaction.append(prod_one_reaction)
    
    # create arrays for the stoichiometry of substrates and products
    sub_stoch = np.zeros((len(one_reaction), total_number_species), int)
    prod_stoch = np.zeros((len(one_reaction), total_number_species), int)
    
    # fill the arrays with the number of species
    for reaction in range(len(sub_number_reaction)):
        for species in sub_number_reaction[reaction]:
            sub_stoch[reaction][species[1]-1] = int(-species[0])
    
    for reaction in range(len(prod_number_reaction)):
        for species in prod_number_reaction[reaction]:
            prod_stoch[reaction][species[1]-1] = int(species[0])
    return(sub_stoch, prod_stoch)

###################################################
######## Gillespie algorithm ######################
###################################################
    
def gillespie_algo(s_i, init, rates, sub_stoch, prod_stoch, tmax, n_max):
    """generates a statistically correct trajectory of a stochastic equation

    input:
    s_i = array([s1,...,sN]) number of slots
    init = array([w1,...,wN,e1,...,eN,p]) number of molecules of each species
    rates = array([c1,..cM]) rates of each reaction
    sub_stoch, prod_stoch = stochiometry of substrates and products in matrix form
    tmax = maximum time
    n_max = estimated maximum number of reactions]

    output:
    store_time = array([[t1],[t2],[t3],...]) current time of each intervall
    store_number_molecules = array([[number molecules reaction 0],[number molecules reaction 0],..])
    store_filling_fraction_av = array([F1,...,FN]) average of filling fraction of each synapse
    coefficient_variation = array([CV_1,...,CV_N]) average of coefficient of variation of each synapse
    """
    
    # ****************************   
    # step 0: initialisation
    # ****************************

    # generate a array of two random numbers for step 2
    r1 = np.random.random_sample(n_max)
    r2 = np.random.random_sample(n_max)

    # initialise constant parameters
    stoch = sub_stoch + prod_stoch 
    number_reactions = np.shape(stoch)[0] # number of reactions
    number_species = np.shape(stoch)[1] # number of species
    number_synapses = int((len(init)-1)/2)

    # initialise current parameters
    current_time = 0
    current_species = init # current number of molecules of each species
    n_counter = 0 # number of already occured reactions
    
    # initialise variables to store time and molecule numbers
    store_time = np.zeros(n_max)
    store_time[n_counter] = current_time
    store_number_molecules = np.zeros((n_max, number_species))
    store_number_molecules[n_counter,:] = current_species 
    store_time_difference = np.zeros((n_max,number_synapses+1))
    store_filling_fraction_av = np.zeros(number_synapses)    
    
    while (current_time < tmax):
        
        # ****************************   
        # step 1: calculate ai and a0
        # ****************************   

        a = np.ones((number_reactions,1))

        for i in range(number_reactions):
            hi = 1  # h1 is defined as the number of distinct 
                    # combinations of Ri reactant molecules 
            for j in range(number_species):
                # check whether the reactant is involved in this reaction
                if sub_stoch[i,j] == 0:
                    continue
                else:
                    # check the reactant has molecules available
                    if current_species[j] <= 0: 
                        hi = 0
                        continue
                    else:
                        hi *= calculate_hi(int(current_species[j]),np.absolute(sub_stoch[i,j]))
                        
            a[i] = hi*rates[i]
            
        a0 = sum(a)

        # ****************************   
        # step 2: calculate the next time difference and reaction
        # ****************************   
        new_time_difference,next_r = next_values(a0,a,r1[n_counter],r2[n_counter])
        store_time_difference[n_counter,:] = np.zeros(number_synapses+1)
        store_time_difference[n_counter,:] += new_time_difference 

        # ****************************   
        # step 3: update the system
        # ****************************   

        # update time, number species, counter
        current_time += new_time_difference 
        current_species += np.transpose(stoch[next_r,:])
        n_counter += 1

        # store current system
        store_time[n_counter] = current_time
        store_number_molecules[n_counter,:] = current_species 
        
    # prepare the final output
    store_time = store_time[:n_counter]
    store_number_molecules = store_number_molecules[:n_counter,:]
    store_time_difference = store_time_difference[:n_counter,:]
    
    # calculate average of filling fraction 
    store_filling_fraction_av = sum(store_time_difference[:,:number_synapses]*store_number_molecules[:,:number_synapses])
    store_filling_fraction_av /= (current_time*s_i)
    
    # calculate average of coefficient of variation
    mol_cv = np.delete(store_number_molecules,7,1)
    mol_cv = np.delete(mol_cv,7,1)
    mol_cv = np.delete(mol_cv,7,1)
    mol_cv = np.delete(mol_cv,7,1)
    mol_cv = np.delete(mol_cv,7,1)
    mol_cv = np.delete(mol_cv,7,1)
    mol_cv = np.delete(mol_cv,7,1)
    average = sum(store_time_difference*mol_cv)
    average /= current_time 
    
    coefficient_variation = np.sqrt(sum((mol_cv - average)**2*(store_time_difference/current_time)))*100/average
    
    print("counter n:", n_counter)
    return(store_time, store_number_molecules, store_filling_fraction_av, coefficient_variation)

###################################################
######## Run the main program #####################
###################################################

# define the stochiometry of the substrates and products
reactions_alpha =\
"X8+X15->X1,X9+X15->X2,X10+X15->X3,X11+X15->X4,X12+X15->X5,X13+X15->X6,X14+X15->X7," 
reactions_beta =\
"X1->X8+X15,X2->X9+X15,X3->X10+X15,X4->X11+X15,X5->X12+X15,X6->X13+X15,X7->X14+X15,"
reactions_delta_gamma = "X15->0X15,0X15->X15"
all_reactions = reactions_alpha + reactions_beta + reactions_delta_gamma
sub_stoch, prod_stoch = \
reactions_stoch(all_reactions) 

# define the initial conditions
F = 0.3 # set F for calculating alpha
s_i = np.array([1,2,5,10,20,50,100]) # set number of slots
e_i = deepcopy(s_i)
s = np.sum(s_i)
beta = 60/43 # set beta
alpha = beta/(2.67*s*(1-0.7)) # set alpha
delta = 1/14 # set delta
gamma = delta*(s*1.0-(beta/alpha)) # set gamma
p = round(gamma/delta) # set p
W = p/1.0 # set number of receptor-slot-complexes in the beginning
tmax = 10 # set the end time 
n_max = 20000 # estimate n_max for later arrays
name_file = "Results"
name_sim = "Simulation AMPAR dynamics"
number_sim = "1"
notes = "Notes: -"
sim_round = 1 # in case of several calculations
times_sim_av = 1 # number of repeated simulations for average

# prepare the rates 
rates = \
np.array([alpha,alpha,alpha,alpha,alpha,alpha,alpha,beta,beta,beta,beta,beta,beta,beta,delta,gamma])

# prepare initial number of molecules of each species according to filling fraction
init1 = (((e_i/s)*W)//1).astype(np.int64)
e_i -= init1
init1 = np.append(init1,e_i)
init1 = np.append(init1,p)
number_synapses = int((len(init1)-1)/2)

# create txt file with important information
name_sim += " " + number_sim
data_in = open(name_file,"a+")
data_in.write(name_sim+"\n")
print(name_sim+"\n")
data_in.write(time.strftime("%d.%m.%Y %H:%M:%S")+"\n")
print(time.strftime("%d.%m.%Y %H:%M:%S"))
data_in.write(notes+"\n")
print(notes+"\n")
data_in.write("\n")
print("")
data_in.write("s_i= "+str(list(s_i))+"\n")
print("s_i= "+str(list(s_i)))
data_in.write("w_i= "+str(list(init1[:number_synapses]))+"\n")
print("w_i= "+str(list(init1[:number_synapses])))
data_in.write("e_i= "+str(list(init1[number_synapses:-1]))+"\n")
print("e_i= "+str(list(init1[number_synapses:-1])))
data_in.write("p= "+str(init1[-1])+"\n")
print("p= "+str(init1[-1]))
data_in.write("alpha = "+str(alpha)+", beta = "+str(beta)+"\n")
print("alpha =",str(alpha)+", beta =",beta)
data_in.write("delta = "+str(delta)+", gamma = "+str(gamma)+"\n")
print("delta = "+str(delta)+", gamma = "+str(gamma))
data_in.write("Chosen Filling Fraction: "+str(F)+" \n")
print("Chosen Filling Fraction: "+str(F))
data_in.write("------------------ ")
data_in.write("\n")
print("------------------")
data_in.write("Results: \n")
print("Results: \n")

# check the result for filling fraction and CV
for r in range(1,sim_round+1):
    ff_av = np.zeros(len(s_i))
    cv = np.zeros(len(s_i)+1)

    counter_simulation = 0
    counter_simulation = 0
    while True:
        init = deepcopy(init1)
        if counter_simulation == times_sim_av:
            break
        results = gillespie_algo(s_i, init, rates, sub_stoch, prod_stoch, tmax, n_max)
        store_filling_fraction_av, coefficient_variation = results[2:]
        store_time = results[0]
        store_molecules = results[1]
        
        # store average filling fraction of simulation
        data_in.write("\n")
        print("\n")
        data_in.write("Simulation "+str(counter_simulation)+"\n")
        print("Simulation "+str(counter_simulation)+"\n")
        data_in.write("F_i: "+str(store_filling_fraction_av)+"\n")
        print("F_i: "+str(store_filling_fraction_av)+"\n")
        ff_av += store_filling_fraction_av
        
        # check whether there is a nan in coefficient_variation
        nan_there = 0
        for i in coefficient_variation:
            if np.isnan(i):
                nan_there = 1
        
        # store average CV of simulation
        data_in.write("CV: "+str(coefficient_variation)+"\n")
        print("CV: "+str(coefficient_variation)+"\n")

        if nan_there == 1:
            continue
        cv += coefficient_variation

        # output the information how many simulations are already done
        counter_simulation += 1
        print("counter simulation: ", counter_simulation)

        # plot and store the number of molecules of the first 4 simulations
        if counter_simulation <= 4: 
            print("")
            print("Simulation "+str(counter_simulation))
            print("Number of molecules of w_i:")
            print("")
            
            # txt file for number of molecules
            w1_data = open("w1_"+str(counter_simulation),"a+")
            w2_data = open("w2_"+str(counter_simulation),"a+")
            w3_data = open("w3_"+str(counter_simulation),"a+")
            w4_data = open("w4_"+str(counter_simulation),"a+")
            w5_data = open("w5_"+str(counter_simulation),"a+")
            w6_data = open("w6_"+str(counter_simulation),"a+")
            w7_data = open("w7_"+str(counter_simulation),"a+")
            p_data = open("p_"+str(counter_simulation),"a+")
            R_data = open("R_"+str(counter_simulation),"a+")
            time_data = open("time_"+str(counter_simulation),"a+")
            
            receptor_array = np.delete(store_molecules,7,1)
            receptor_array = np.delete(receptor_array,7,1)
            receptor_array = np.delete(receptor_array,7,1)
            receptor_array = np.delete(receptor_array,7,1)
            receptor_array = np.delete(receptor_array,7,1)
            receptor_array = np.delete(receptor_array,7,1)
            receptor_array = np.delete(receptor_array,7,1)

            for i in range(len(store_molecules)):
                w1_data.write(str(store_molecules[i,0])+"\n")
                w2_data.write(str(store_molecules[i,1])+"\n")
                w3_data.write(str(store_molecules[i,2])+"\n")
                w4_data.write(str(store_molecules[i,3])+"\n")
                w5_data.write(str(store_molecules[i,4])+"\n")
                w6_data.write(str(store_molecules[i,5])+"\n")
                w7_data.write(str(store_molecules[i,6])+"\n")
                p_data.write(str(store_molecules[i,-1])+"\n")
                R_data.write(str(sum(receptor_array[i]))+"\n")
                time_data.write(str(store_time[i])+"\n")
            w1_data.close()
            w2_data.close()
            w3_data.close()
            w4_data.close()
            w5_data.close()
            w6_data.close()
            w7_data.close()
            p_data.close()
            R_data.close()
            time_data.close()
            
        data_in.write("End run time: "+time.strftime("%d.%m.%Y %H:%M:%S")+"\n")
        print("End run time: "+time.strftime("%d.%m.%Y %H:%M:%S"))

    # calculate the average of F and CV of all simulations
    ff_av /= times_sim_av
    cv /= times_sim_av

    # store CV and F in txt file
    cv_ff_data = open("cv_ff","a+")
    cv_ff_data.write("CV\n")
    for i in range(len(cv)):
        cv_ff_data.write(str(cv[i])+"\n")
    cv_ff_data.write("F\n")
    for i in range(len(ff_av)):
        cv_ff_data.write(str(ff_av[i])+"\n")
    cv_ff_data.close()
    
    # output and save the results in the txt file
    data_in.write("\n")
    print("\n")
    data_in.write("Number of repeated simulations: "+str(counter_simulation)+"\n")
    print("Number of repeated simulations: "+str(counter_simulation))
    data_in.write("Values for synapse 1,2,3,..:\n")
    print("Values for synapse 1,2,3,..:")
    data_in.write("F average: "+str(list(np.round(ff_av, 2)))+"\n")
    print("F average:", list(np.round(ff_av, 2)))
    data_in.write("Coefficient of Variation average: "+str(list(np.round(cv, 2)))+"\n")
    print("Coefficient of Variation average:", list(np.round(cv, 2)))
    

##########################################################
#plot w_i of each synapse during simulation: Fig. 8A,B ###
##########################################################

# extract the information from txt files
sim = "1"
wi = [0,0,0,0,0,0,0]
for species in range(7):
    wi_data = open("w"+str(species+1)+"_"+sim)
    wi[species] = np.array([])
    
    for line in wi_data:
        wi[species] = np.append(wi[species], float(line.rstrip()))
    wi_data.close()

R = np.array([])
data = open("R"+"_"+sim)
for line in data:
    R = np.append(R, float(line.rstrip()))
data.close()

p = np.array([])
data = open("p"+"_"+sim)
for line in data:
    p = np.append(p, float(line.rstrip()))
data.close()

time = np.array([])
data = open("time"+"_"+sim)
for line in data:
    time = np.append(time, float(line.rstrip()))
data.close()

# plot
f = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.rc('text', usetex=True)

color_lines = ['coral','black','darkviolet','red','cyan','green','blue']
labels = [r'$1$', r'$2$', r'$5$',r'$10$', r'$20$', r'$50$',r'$100$']
for i in [6,5,4,3,2,1,0]:
    plt.plot(time, wi[i],'k',color=color_lines[i],label=labels[i],linewidth=0.3)
plt.xlabel(r'$t \; [{\rm min}]$', fontsize=12)
plt.ylabel(r'$w_i$', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.title(r'$F=0.5$', fontsize=12)
plt.show()
f.savefig("plot_wi_Fig8AB.pdf", bbox_inches='tight')

###################################################
######## plot CV ##################################
###################################################

# this is an exemplary plot with one measurement of CV
# in file coefficientOfVariation.py you can find a plot that combines different CV's of multiple simulations into one figure

# extract the information from txt files
cv = np.array([])
data = open("cv_ff")
counter = 0
for line in data:
    counter += 1
    if counter == 1:
        continue
    if counter <= 9:
        cv = np.append(cv, float(line.rstrip())) 
    if counter == 10:
        break
data.close()

# calculate regression
x_axis = np.array([1,2,5,10,20,50,100])*0.7 # in case of F=0.7

av_x_axis = sum(np.log(x_axis))/len(x_axis)

av_cv = sum(np.log(cv[:-1]))/len(cv[:-1])

b_cv=sum((np.log(x_axis)-av_x_axis)*(np.log(cv[:-1])-av_cv))/sum((np.log(x_axis)-av_x_axis)**2)
a_cv=av_cv-b_cv*av_x_axis

X = np.linspace(0.4,100,500,endpoint=True)
Y_cv = np.exp(a_cv)*(X**b_cv)

# plot
f = plt.figure(figsize=(4,3))
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
plt.rc('font', serif='Times New Roman') 
plt.rc('text', usetex=True)

plt.xscale('log')
plt.yscale('log')
plt.plot(x_axis, cv[:-1],'.',color='blue',label=r'$F=0.7$')
plt.plot(X, Y_cv,'-',color='blue',alpha = 0.3)
plt.ylabel(r'$CV \; [{\rm \%}]$', fontsize=12)
plt.xlabel(r'$Fs_i$', fontsize=12)
#plt.title(r'$p=2.67$', fontsize=12)
plt.legend(loc=1, fontsize=12)
plt.show()
f.savefig("plot_cv_Fig8CD.pdf", bbox_inches='tight')
