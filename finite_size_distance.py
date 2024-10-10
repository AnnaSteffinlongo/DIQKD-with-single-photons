
"""
Script to compute the finite size key rate as a function of the distance.
This script in particular reproduces Fig. 3.
"""

def load_params(namefile):  
    """
    Opens a cvs file with the noisy preprocessing parameters and loads it. Outputs the vectors with the data
    
        namefile    --    name of the file from which to load the data
    """
    csv_data = []
    with open(namefile, 'r') as file:

        csv_reader = csv.reader(file)

        # Iterate over the rows and append them to the list
        for line in csv_reader:
            csv_data += [line[1:]]
                       
    l= len(csv_data[0])
    N = [float(csv_data[0][x]) for x in range(l)]
    keys = [float(csv_data[1][x]) for x in range(l)]
    
    return  N, keys

import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# value of the transmissivity T
t=0.005

# Load data - in this example we use T=0.005 and eta_L=0.9
data = './../New_noise/t' + str(t) + '_eta_T0.9_finite_size.csv'
data_pol = './../polarization/Finite_size/eta_T0.9pol_finite_size.csv'
N, keys = load_params(data)
N_pol, keys_pol = load_params(data_pol)

# Asymptotic key rate
asymptotic = keys[0]
asymptotic_pol = keys_pol[0]

# Compute heralding probability for ideal and realistic central detectors
etaD=0.8
length = np.logspace(0,3,1000)
etaC = np.sqrt(10**(-length*(0.2/10)))

# Heralding probability for polarization scheme
PH_pol = t**2* etaD**2* etaC**2 
PH_ID_pol = t**2* etaC**2 

# Heralding probability for single-photon scheme
PH = t* etaD* etaC
PH_ID = t* etaC

# Single photon generation frequency
freq = 5*1e6

# Elements to plot (each element corresponds to the key rate for a certain N)
elem_list_pol = [9,20,22,23]
elem_list = [10,17,19,20]

colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

# Color definition
num_shades = 5
dark_red = '#FF5733'
light_red = '#FFB999'
red_colors = [plt.cm.Reds(i) for i in np.linspace(1, 0.3, num_shades)]

dark_blue = '#3366FF'
light_blue = '#99A3FF'
blue_colors = [plt.cm.Blues(i) for i in np.linspace(1, 0.3, num_shades)]

red_handles, red_labels = [], []
blue_handles, blue_labels = [], []

fig, ax = plt.subplots()

# Prepare plot for our sceheme
for col, elem in zip(red_colors, elem_list):
    key_rate_ID = PH_ID * keys[elem] * freq
    key_rate = PH * keys[elem] * freq

    red_handle, = plt.plot(length, key_rate_ID, color=col, linewidth=2)
    plt.plot(length, key_rate, color=col, linewidth=2, linestyle='--')
    red_handles.append(red_handle)
    red_labels.append('N=' + f"{N[elem]:.1e}")
    
# Prepare plot for polarization scheme 
for col, elem_pol in zip(blue_colors, elem_list_pol):
    key_rate_ID_pol = PH_ID_pol * keys_pol[elem_pol] * freq
    key_rate_pol = PH_pol * keys_pol[elem_pol] * freq

    blue_handle, = plt.plot(length, key_rate_ID_pol, color=col, linewidth=2)
    plt.plot(length, key_rate_pol, color=col, linewidth=2, linestyle='--')
    blue_handles.append(blue_handle)
    blue_labels.append('N=' + f"{N_pol[elem_pol]:.1e}")

# Legend definition
red_legend = plt.legend(red_handles, red_labels, loc='upper right', prop={'family': 'serif', 'size': 9})
ax.add_artist(red_legend)
blue_legend = plt.legend(blue_handles, blue_labels, loc='center right', prop={'family': 'serif', 'size': 9})


plt.xlabel("L (km)", fontdict={'family': 'serif', 'size': 12})
plt.ylabel("R (bits/s)", fontdict={'family': 'serif', 'size': 12})
plt.xscale('log')
plt.yscale('log')
plt.xticks(fontfamily='serif', fontsize=12)
plt.yticks(fontfamily='serif', fontsize=12)
plt.ylim(1e-1, 2e3)
plt.xlim(length[0], 1e3)

# plt.savefig('./../Final_plots/Distance_5MHz_etaT0.9.pdf')



