"""
Given the relation:
G(\tau,k) = \int K(\omega, \tau) * A(\omega,k) d\omega
and knowing K and G, we solve for A. K is a known kernel for fermionic 
(electronic spectral functions) or bosonic (structure factors) 



This file will take arguments passed via command line to perform analytic continuation

    python3 ana_cont.py FOLDER CORRELATION SPACE OMEGA_MIN OMEGA_MAX N_OMEGA K_MIN K_MAX ORBITALS ALPHA OPTIMIZER NAMETAG

examples:
1D Chain, k points [0,19], treat each orbital separately
    python3 ana_cony.py "/path_to/hubbard_cuprate_n1.00_n-1.450-1" "spin_z" "momentum"\
                         -10.0 10.0 101 0 19 all "chi2kink" "scipy_lm" "chi2run"

1D Chain, default settings, all k points
    python3 ana_cony.py "/path_to/hubbard_cuprate_n1.00_n-1.450-1" "greens_up" "position"\
                         -10.0 10.0 101

3D, passing 3D k points with k_x = 1, k_y = [0,19], k_z = 4, merge orbitals 0 with 2, 1 with 3
    python3 ana_cony.py "/path_to/hubbard_cuprate_n1.00_n-1.450-1" "spin_z" "momentum"\
                         -10.0 10.0 101 1x0x4 1x19x4 0+2x1+3

                    
                     


FOLDER:
    Output folder for SmoQyDQMC
CORRELATION:
    'spin_z', 'density', 'greens_up', 'greens_dn' or 'phonon_greens'
SPACE:
    'position' or 'momentum'
OMEGA_MIN, OMEGA_MAX, N_OMEGA:
    If you only apply Analytic Continuation on the range you care about plotting 
    it is highly likely to fail! If that happens, you may need to expand your range
    of omegas and trim the data in post-processing
    E.g. for density omega = [-10.0, 10.0] n=201 has given me good results. Same for 
    spin_z, where I trim the data to [0.0, 2.0] later
K_MIN, K_MAX (optional): Default is "all"
    K points to include (Python nomenclature). Dimensions are separated by "x"
    alternatively you may use "all" "all" to parse all k points 
ORBITALS (optional): default is "all"
    Orbitals to merge prior to computing AC. Default is to do all of them separately. Separate combinations are  You can instead
    merge orbitals as you wish. E.g. for 4 orbitals, to do orbital 1 alone and merge 0, 2, and 3
    you use "1x0+2+3". To do just 0,1, and 3 separately you do "0x1x3". "all" will do all orbitals alone  
ALPHA: (optional): Default is 'chi2kink'
    Algorithm for calculating alphas in max ent method. Options are 'chi2kink', 'historic',
    'classic', and 'bryan'. AFAIK bryan is broken. 'chi2kink' is recommended.
OPTIMIZER: (optional): Default is 'scipy_lm'
    Optimizer for finding ideal alpha. Options are 'newton' and 'scipy_lm'. 'newton' is much faster,
    but is much more likely to fail
NAMETAG: (optional): Default is empty string ''
    This is appended to output file names. Useful if you want to run multiple ALPHA algorithms
    on the same data for comparison. For NAMETAG = "HelloWorld" an output file of 
    'spin_z_k.csv' would instead be named 'spin_z_k_HelloWorld.csv'

Data files need to be the collapsed (unbinned) version
In SmoQyDQMC that means you need to run something along the lines of
    
    folder_string = "hubbard_cuprate_n1.00_n-1.450-1"
    process_correlation_measurement(folder=folder_string,correlation="density",
                                    type="time-displaced",space="momentum",N_bin=100)
    

OUTPUT:
Output files will be space delimited csv files in a new subdirectory 'ana_cont' of the directory passed in the 
first command line argument. The file name will be 'CORRELATION_oORBITAL_k1_k2_k3.csv' or 
'CORRELATION_oORBITAL_k1_k2_k3_NAMETAG.csv' if you added a NAMETAG. E.g. the file will be 
/path_to/hubbard_cuprate_n1.00_n-1.450-1/ana_cont/spin_z_o3_2_0_0.csv
For consistency three k values will always be used, hopefully making post processing easier

"""

import numpy as np
import sys
import os
import pandas as pd
import toml
sys.path.insert(0, "/home/james/Documents/code/ana_cont") # James' laptop
sys.path.insert(0, "/lustre/isaac/proj/UTK0014/AnalyticContinuation/ana_cont") # ISAAC NG
import ana_cont.continuation as cont
"""
ana_cont citation.
Package bibtex:
@article{KaufmannJosef2023aPpf,
author = {Kaufmann, Josef and Held, Karsten},
copyright = {2022 The Authors},
issn = {0010-4655},
journal = {Computer physics communications},
keywords = {Analytic continuation ; Maximum entropy ; Padé},
language = {eng},
pages = {108519-},
publisher = {Elsevier B.V},
title = {ana_cont: Python package for analytic continuation},
volume = {282},
year = {2023},
}
"""


# This function will get the relevant settings from the MC run 
def parse_toml(folder):
    toml_str = folder + "/model_summary.toml"
    try:
        toml_dict = toml.load(toml_str)
    except:
        print("model_summary.toml not found.\nPerhaps there's a typo in the folder path or the simulation did not finish")
        exit()
    n_dims = int(toml_dict['geometry']['dimensions'])
    beta = float(toml_dict['beta'])
    n_tau = int(toml_dict['L_tau'])+1 #+1 to include tau=0
    n_orbitals = int(toml_dict['geometry']['unit_cell']['orbitals'])
    k_max = np.asarray(toml_dict['geometry']['lattice']['L']) - 1 # Python 0 indexing
    k_min = np.zeros_like(k_max)   
    return n_dims, beta, n_tau, n_orbitals, k_min, k_max 

# Parse arguments
num_args = len(sys.argv)
if num_args < 7:
    print("Required arguments are \n\tFOLDER CORRELATION SPACE OMEGA_MIN OMEGA_MAX N_OMEGA\nExiting")
    exit()

folder_str = sys.argv[1]
n_dims, beta, n_tau, n_orbitals, k_min_run, k_max_run = parse_toml(folder_str)

# load correlation string, correct for likely typos
correlation_str = sys.argv[2].lower().replace('-','_')
allowed_correlations = ('spin_z', 'density', 'greens_up', 'greens_dn', 'phonon_greens',"spin_x")

if allowed_correlations.count(correlation_str) == 0:
    print(correlation_str," is not an allowed correlation")
    exit()
if (correlation_str == 'greens_up') or (correlation_str == 'greens_dn'):
    kernel_str = 'time_fermionic'
else:
    kernel_str = 'time_bosonic'

allowed_spaces = ('momentum','position')
space_str = sys.argv[3].lower()
if allowed_spaces.count(space_str) == 0:
    print(space_str," is not allowed. Use momentum or position, only")


omega_min = float(sys.argv[4])
omega_max = float(sys.argv[5])
omega_num = int(sys.argv[6])
omega_step = (omega_max - omega_min) / (omega_num - 1)

# K_MIN and K_MAX
if num_args > 8:
    k_min_str = sys.argv[7]
    if (k_min_str != "all"):
        k_min_str_array = k_min_str.split('x')
        k_min = [int(i) for i in k_min_str_array]
        k_max_str_array = sys.argv[8].split('x')
        k_max = [int(i) for i in k_max_str_array]
        if (np.shape(k_min)[0] != n_dims) or (np.shape(k_max)[0] != n_dims):
            print("Simulation was",n_dims,"dimensional. Incorrect dimensions on K values")
            exit()
    else:
        print("Defaulting to calculating function for all k points")
        k_max = k_max_run
        k_min = k_min_run
else:
    print("Defaulting to calculating function for all k points")
    k_max = k_max_run
    k_min = k_min_run
# ORBITALS
if num_args > 9:
    if sys.argv[9].lower() == "all":
        orbital_list = np.zeros((n_orbitals,1),int)
        for orb in range(0,n_orbitals):
            orbital_list[orb] = orb
    else:
        try:
            txt_groups = sys.argv[9].split('x')
            n_orbital_groups = len(txt_groups)
            maxlen = 1
            for i in range(0,n_orbital_groups):
                maxlen = max(maxlen,1+txt_groups[i].count('+'))
            orbital_list = np.full((n_orbital_groups,maxlen),-1,int)
            for group in range(0,n_orbital_groups):
                split_string = txt_groups[group].split('+')
                for orb in range(0,len(split_string)):
                    orbital_list[group,orb] = int(split_string[orb])
        except:
            print(sys.argv[9]," is not an acceptable input for ORBITALS")
            exit()
else:
    # default to all  
    orbital_list = np.zeros((n_orbitals,1),int)
    for orb in range(0,n_orbitals):
        orbital_list[orb] = orb

# ALPHA
if num_args > 10:
    alpha_method = sys.argv[10].lower()
    allowed_alphas = ('historic','classic','bryan','chi2kink')
    if allowed_alphas.count(alpha_method) == 0:
        print(alpha_method,"is not an allowed ALPHA.")
        print("Only \'chi2kink\', \'historic\', \'classic\', and \'bryan\' are allowed")
        exit()
else:
    alpha_method = 'chi2kink'
# OPTIMIZER
if num_args > 11:
    optimizer = sys.argv[11].lower()
    allowed_optimizers = ('newton','scipy_lm')
    if allowed_optimizers.count(optimizer) == 0:
        print(optimizer, "is not an allowed OPTIMIZER")
        print("Only \'newton\' or \'scipy_lm\' are allowed")
        exit()
else:
    optimizer = 'scipy_lm'
# NAMETAG
if num_args > 12:
    nametag = '_' + sys.argv[12]
else:
    nametag = ''
data_file_str = folder_str + '/time-displaced/' + correlation_str + '/' + correlation_str + "_" +space_str +"_time-displaced_stats.csv"

# Using pandas read the data file 
try:
    data_frame = pd.read_csv(data_file_str,na_filter=False,delimiter=' ')   
except:
    print(data_file_str,"not found")
    print("You may need to run process_correlation_measurement(...) in Julia")
    exit()
data_values_tmp = pd.DataFrame(data_frame, columns=['mean_r']).to_numpy()
data_err_tmp = pd.DataFrame(data_frame, columns=['std']).to_numpy()
if correlation_str == "phonon_greens":
    data_values_tmp = - data_values_tmp

# Sometimes the code outputs diagonal elements first, sometimes off diagonals
# Find correct ranges for diagonal elements
data_ID1 = pd.DataFrame(data_frame, columns=['ID_1']).to_numpy()
data_ID2 = pd.DataFrame(data_frame, columns=['ID_2']).to_numpy()
# Calculate total number of k points per orbital
k_total = 1
for dim in range(0,n_dims):
    k_total *= k_max_run[dim]+1
# Calculate length of each ID pair dataset
ID_t_k_len = int(n_tau * k_total)
total_ID_pairs = int(len(data_ID1)/ID_t_k_len)
# fill final data
data_values = np.zeros((n_orbitals*ID_t_k_len,1),np.double)
data_err = np.zeros((n_orbitals*ID_t_k_len,1),np.double)
for pair in range(0,total_ID_pairs):
    low_pos_tmp = int(pair*ID_t_k_len)
    if data_ID1[low_pos_tmp] == data_ID2[low_pos_tmp]:
        high_pos_tmp = low_pos_tmp + ID_t_k_len
        low_pos = int((data_ID1[low_pos_tmp]-1) * ID_t_k_len)
        high_pos = int(low_pos + ID_t_k_len)
        data_values[low_pos:high_pos] = data_values_tmp[low_pos_tmp:high_pos_tmp]
        data_err[low_pos:high_pos] = data_err_tmp[low_pos_tmp:high_pos_tmp]


# Enumerate omega and tau values
omega_vals = np.linspace(omega_min,omega_max,num=omega_num,endpoint=True)
tau_vals = np.linspace(0.0,beta,num=n_tau,endpoint=True)

# Omega = 0.0 as a value is frequently problematic, fix for it
omega_vals[omega_vals == 0.0] = omega_step/10.0

# Reshape arrays to [n_tau, n_orbitals, nk_1, nk_2, nk_3]
# There is likely a savvier way of doing this, but meh
if (n_dims==1):
    k_shape = [k_max_run[0]+1,1,1]
elif (n_dims==2):
    k_shape = [k_max_run[0]+1,k_max_run[1]+1,1]
else:
    k_shape = [k_max_run[0]+1,k_max_run[1]+1,k_max_run[2]]

k_max_adj = np.ones((3),int)
k_min_adj = np.zeros((3),int)
k_max_adj[0:n_dims] = np.array(k_max[0:n_dims]) +1
k_min_adj[0:n_dims] = k_min[0:n_dims]

data_values = np.reshape(data_values, (n_orbitals,n_tau,k_shape[0],k_shape[1],k_shape[2]))
data_err = np.reshape(data_err, (n_orbitals,n_tau,k_shape[0],k_shape[1],k_shape[2]))

# Create flat Bayesian prior
model = np.ones_like(omega_vals)
model /= np.trapz(model,omega_vals)

# Create output folder if needed
if (not os.path.exists(folder_str + "/ana_cont")):
    os.mkdir(folder_str + "/ana_cont")



num_failure = 0.0
num_try = 0.0
# Loop over all orbital groupings and k points
for orbital_group in range(0,np.shape(orbital_list)[0]):
    for k1 in range(k_min_adj[0],k_max_adj[0]):
        for k2 in range(k_min_adj[1],k_max_adj[1]):
            for k3 in range(k_min_adj[2],k_max_adj[2]):
                output = np.zeros((omega_num,2))
                output[:,0] = omega_vals
        
                for orbital in orbital_list[orbital_group]:
                    # If list is done values of -1 will break from this loop
                    write = True
                    if orbital == -1:
                        break    
                    num_try += 1.0
                    # Set up AC
                    probl = cont.AnalyticContinuationProblem(im_axis=tau_vals, re_axis=omega_vals,
                                                            im_data=data_values[orbital,:,k1,k2,k3],
                                                            kernel_mode=kernel_str, beta=beta)
                    try:
                        sol,_ = probl.solve(method='maxent_svd',
                                            alpha_determination=alpha_method,
                                            optimizer=optimizer,
                                            stdev=data_err[orbital,:,k1,k2,k3], model=model,
                                            interactive=False,
                                            verbose=False)
                        output[:,1] += sol.A_opt
                        print(np.count_nonzero(output[:,1]))
                    except:
                        print("Solver failed to converge for")
                        print("   orbital:",orbital,"k1:",k1,"k2:",k2,"k3:", k3)
                        num_failure += 1.0
                        write = False
                    if write:    
                        output_file_name = folder_str + '/ana_cont/' + correlation_str + '_' + space_str+ '_o' + txt_groups[orbital_group] + '_' + str(k1) + '_' + str(k2) + '_' + str(k3) + nametag + '.csv'
                        np.savetxt(output_file_name,output,delimiter=' ')
print("Code finished with",int(num_failure),"of",int(num_try), "attempts failing")
print("Failure rate =",num_failure/num_try)
exit()

