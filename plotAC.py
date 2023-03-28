#!/bin/python3

import numpy as np
import os
import toml
import matplotlib.pyplot as plt
import pandas as pd

class GenericError(Exception):
   
    # Constructor or Initializer
    def __init__(self, error_type, data) -> None:
        self.error_type = error_type
        self.data = data
   
    # __str__ is to print() the value
    def __str__(self):
        return(repr(self.data))
    

# plotAC
# This class handles SmoQyDQMC output which has been process through ana_cont.py
# It incorporates features to quickly plot spectral functions and dynamic structure factors
class plotAC:
    # List of allowed correlation functions
    _allowed_correlations = ('spin_z', 'density', 'greens_up', 'greens_dn', 'phonon_greens', 'spin_x', 'pair')
    _allowed_spaces = ('momentum','position')

    # Constructor
    #  sim_directory    : root folder for a simulation
    #  output_directory : folder to put images and combined data files
    #                     defaults to sim_directory
    #  TO BE ADDED: nametag : if your files have a nametag inserted before '.csv'
    def __init__(self,sim_directory,output_directory=None,nametag="") -> None:
        self._directory = sim_directory
        if output_directory == None:
            self._out_directory = self._directory
        else:
            self._out_directory = output_directory
            if (not os.path.exists(output_directory)):
                os.mkdir(output_directory)
        self.__parse_toml()
        self._nametag = nametag
        if nametag != "":
            self._nametag = "_" + nametag
        
        return None

    # __parse_toml
    #  Loads data from a simulation's model_summary.toml
    def __parse_toml(self) -> None:
        toml_str = self._directory + "/model_summary.toml"
        try:
            toml_dict = toml.load(toml_str)
        except:
            print("model_summary.toml not found.\nPerhaps there's a typo in the folder path or the simulation did not finish")
            exit()
        self._n_dims = int(toml_dict['geometry']['dimensions'])
        self._beta = float(toml_dict['beta'])
        self._n_tau = int(toml_dict['L_tau'])+1 #+1 to include tau=0
        self._n_orbitals = int(toml_dict['geometry']['unit_cell']['orbitals'])
        self._n_k = np.zeros((3),int)
        n_k_tmp = np.asarray(toml_dict['geometry']['lattice']['L']) 
        self._n_k[0:self._n_dims] += n_k_tmp[0:self._n_dims]
        self._n_k_loop = np.copy(self._n_k)
        self._n_k_loop[self._n_k_loop<1] = 1
        self._total_k = self._n_k_loop[0] * self._n_k_loop[1] *self._n_k_loop[2] 
        return None
    
    # Returns model parameters from a simulation:
    #    n_dims, beta, n_tau, n_orbitals, n_k
    #  n_k is a 1D array of len 3
    def get_model_parameters(self):
        return self._n_dims, self._beta, self._n_tau, self._n_orbitals, self._n_k

    # Returns list of allowed correlations
    def get_allowed_correlations(self):
        return self._allowed_correlations

    # Returns list of allowed spaces
    def get_allowed_spaces(self):
        return self._allowed_spaces
    

    # Checks if entire data set exists for a correlation type and orbital
    #   and if the omegas are the same for all files
    # correlation: string e.g. "greens_up", etc
    # space      : "momentum" or "position"
    # orbital    : string in "0+3" format
    #  
    # Returns True/False 
    def check_data_exists(self,correlation,space,orbital):
        found = True
        
        if self._allowed_correlations.count(correlation) == 0:
            print("Error, correlation ",correlation," not supported.")
            found = False
        else:
            if self._allowed_spaces.count(space)==0:
                print("Error, space ", space, " not supported")
            else:
                try:
                    for k1 in range(0,self._n_k_loop[0]):
                        for k2 in range(0,self._n_k_loop[1]):
                            for k3 in range(0,self._n_k_loop[2]):
                                file_name = correlation +'_' +space +"_o" + str(orbital) + '_' + str(k1) + '_' + str(k2) + '_' + str(k3) + self._nametag+'.csv'
                                if not os.path.isfile(self._directory + "/ana_cont/" + file_name):
                                    raise(GenericError("FilesMissing",file_name))
                                tmp_dat = np.loadtxt(self._directory+ "/ana_cont/" + file_name,delimiter=' ')
                                if (k1==0 and k2==0 and k3 ==0):
                                    omegas = tmp_dat[:,0]
                                else:
                                    if not np.array_equal(omegas,tmp_dat[:,0]):
                                        raise(GenericError("BadOmegas",(k1,k2,k3)))
                except GenericError as error:
                    found = False
                    if (error.error_type == "FilesMissing"):
                        print(error.data + " not found. Returning False")            
                    elif (error.error_type == "BadOmegas"):
                        print("Omegas for k's ",error.data," do not equal those of (0,0,0)")
                    
        return found
    
    # Merges data from ana_cont.py output and returns 4D array [omegas,k1,k2,k3]
    # correlation: string "greens_up", etc
    # space      : string "momentum" or "position"
    # orbital    : string in "0+2" format
    # save_merged: after merging the data save it to output folder
    #
    # Returns 4D array [omegas,k1,k2,k3]
    def merge_data(self,correlation,space,orbital,save_merged=False):
        success = True
        # Load omega list
        try:
            file_name = correlation + '_' + space  + "_o" + str(orbital) + '_0_0_0.csv'
            omegas = np.loadtxt(self._directory + "/ana_cont/" + file_name,delimiter=' ')[:,0]
        except:
            print("Couldn't load omegas from ",self._directory + "/ana_cont/" + file_name)
            print("try using check_data_exists method first")
            return None,None,success
        # Data
        data_arr = np.zeros((len(omegas),self._n_k_loop[0],self._n_k_loop[1],self._n_k_loop[2]))
        
        try:             
            for k1 in range(0,self._n_k_loop[0]):
                for k2 in range(0,self._n_k_loop[1]):
                    for k3 in range(0,self._n_k_loop[2]):
                        file_name = correlation +'_' +space  + "_o" + str(orbital) + '_' + str(k1) + '_' + str(k2) + '_' + str(k3) + '.csv'
                        tmp_dat = np.loadtxt(self._directory+ "/ana_cont/" + file_name,delimiter=' ')
                        data_arr[:,k1,k2,k3] = tmp_dat[:,1]
     
        except:
            print("Error in loading data for ",correlation)
            success = False
            return omegas,data_arr,success

        if save_merged==True:
            self.save_data(correlation,orbital,data_arr,omegas)
        return omegas,data_arr,success

    # Saves the data in csv format to output directory
    # correlation: string "greens_up", etc
    # space      : string "momentum" or "position"
    # orbital    : string in "0+2" format
    # data_array : 4D array of shape (n_omegas,k1,k2,k3)
    # omegas     : 1D array with the value of each omega point
    # 
    # Returns nothing 
    def save_data(self,correlation,space,orbital,data_array,omegas):
        n_omega = len(omegas)
        
        data_array_flat = data_array.flatten()
        # omegas
        data_omegas = np.zeros((len(data_array_flat)),np.double)
        for i in range(0,n_omega):
            data_omegas[i*self._total_k:(i+1)*self._total_k] = omegas[i]
        # k1
        data_k_1_tmp = np.zeros(self._total_k,int)
        k1_split = self._n_k_loop[1]*self._n_k_loop[2]
        for i in range(0,self._n_k_loop[0]):
            data_k_1_tmp[i*k1_split:(i+1)*k1_split] = i
        data_k_1 = np.tile(data_k_1_tmp,n_omega)
        # k2
        data_k_2_tmp = np.zeros(self._n_k_loop[1]*self._n_k_loop[2],int)
        for i in range(0,self._n_k_loop[1]):
            data_k_2_tmp[i*self._n_k_loop[2]:(i+1)*self._n_k_loop[2]] = i
        data_k_2 = np.tile(data_k_2_tmp,n_omega*self._n_k_loop[0])
        # k3
        data_k_3 = np.tile(np.arange(self._n_k_loop[2]),n_omega*self._n_k_loop[0]*self._n_k_loop[1])
        csv_data = {
            "OMEGA"     : data_omegas,
            "K_1"       : data_k_1,
            "K_2"       : data_k_2,
            "K_3"       : data_k_3,
            correlation : data_array_flat,
        }
        df = pd.DataFrame(csv_data)
        df.to_csv(self._out_directory + "/" + correlation + '_' +  space +   "_o"+str(orbital)+ ".csv", sep=' ')
        return
    
    # Loads data from save_data method
    # correlation: string "greens_up", etc
    # space      : string "momentum" or "position"
    # orbital    : string in "0+2" format
    # 
    # Returns 1D array of omega values, 4D array of saved values (n_omega,k1,k2,k3) 
    def load_data(self,correlation,space,orbital):
        data_file_str = self._out_directory +"/" + correlation +"_o"+str(orbital)+ ".csv"
        df = pd.read_csv(data_file_str,delimiter=' ')
        omegas = pd.DataFrame(df, columns=['OMEGA']).to_numpy()
        omegas = omegas.reshape((-1,self._total_k))[:,0]
        data_array = pd.DataFrame(df, columns=[correlation]).to_numpy()
        data_array = data_array.reshape((len(omegas),self._n_k_loop[0],self._n_k_loop[1],self._n_k_loop[2]))
        return omegas,data_array

    # Modifies 2D data with a host of capabilities
    # data_array_2D : shape of (len(omegas),n_k)
    # omegas        : list of omega values
    #
    # This modifies the data passed in with the following operations:
    # omega_min : trim values below an energy threshold
    # omega_max : trim values above an energy threshold
    # center_k0 : move k axis (a=1) from [0,2pi) to (-pi,pi]
    # zero_k0   : zeros out k=0 data (useful for susceptibilities where AC has trouble at k=0) 
    # trim_pi   : Removes outer most k value which may wash out data. Requires center_k0 = True and even n_k
    # duplicate_pi : makes sure -pi = pi. Requires center_k0 = True and even n_k. mutually exclusive with trim_pi
    def modify_data(self,data_array_2D,omegas,omega_min=None,omega_max=None,center_k0=False,zero_k0=False,trim_pi=False,duplicate_pi=False):
        tiny = 1e-4 #omegas sometimes get a little off when loading from files, allow some fudge factor
        data_array_new = np.array(data_array_2D)
        omegas_new = omegas
        # trim omegas
        if omega_min != None:
            data_array_new = np.array(data_array_new[omegas_new>=omega_min-tiny,:])
            omegas_new = omegas_new[omegas_new>=omega_min-tiny]
        if omega_max != None:
            data_array_new = np.array(data_array_new[omegas_new<=omega_max+tiny,:])
            omegas_new = omegas_new[omegas_new<=omega_max+tiny]
        extent = 0, 2, np.min(omegas_new), np.max(omegas_new)
        # set k0=0
        if zero_k0 == True:
            data_array_new[0,:] = 0.0
        # adjust to set k=0 in middle of x axis
        if center_k0 == True:
            n_k = data_array_new.shape[1]
            n_k_2 = int((n_k+1)/2)
            permutation = np.zeros((n_k),int)
            for k in range(0,n_k):
                permutation[k] = (k+n_k_2)%n_k
            
            data_array_new = data_array_new[:,permutation]
     
            if n_k%2==0:
                extent_mod = float(n_k_2 -1)/float(n_k)
                extent = -1.0, extent_mod, np.min(omegas_new), np.max(omegas_new)
                if trim_pi == True:
                    data_array_new = data_array_new[1:,:]
                    extent = -extent_mod, extent_mod, np.min(omegas_new), np.max(omegas_new)
                elif duplicate_pi == True:
                    data_array_tmp = np.zeros((data_array_new.shape[0]+1,data_array_new.shape[1]))
                    data_array_tmp[:data_array_new.shape[0],:] = data_array_new[:,:]
                    data_array_new = data_array_tmp
                    data_array_new[-1,:] = data_array_new[0,:]
        return data_array_new, omegas_new, extent

    # Takes full 4D array (omegas,k1,k2,k3) and returns 2D array (omegas, n_k)
    #   You can take a cut along a single axis or a diagonal using this method
    #   To do diagonals the axis must be of the same number of k points!
    # data_array   : data in shape (omegas,k1,k2,k3)
    # k_points_low : starting k points when making a cut
    # k_points_high: ending k points for a cut
    # ascending    : True means it takes cut from low k point to high in that dimension. False means
    #                you take the cut in descending order (like M->Gamma)
    def make_1D(self,data_array,k_points_low=(0,0,0),k_points_high=(-1,0,0),ascending=(True,True,True)):
        # I can spend a few hours figuring out how to do this in a cute way
        # instead let's brute force this guy
        k1 = k_points_low[0]!=k_points_high[0]
        k2 = k_points_low[1]!=k_points_high[1]
        k3 = k_points_low[2]!=k_points_high[2]
        for i in range(0,3):
            if k_points_high[i] == -1:
                k_points_high[i] = data_array.shape[i+1]
        if not (k1 or k2 or k3):
            print("All axes false, returning garbage")
            return None
        k_length = int(0)
        if k1:
            k_length = k_points_high[0]-k_points_low[0]
        if k2:
            if k_length == 0:
                k_length = k_points_high[1]-k_points_low[1]
            elif k_length != k_points_high[1]-k_points_low[1]:
                print("Axis 2 does not match Axis 1 in length")
                return None
        if k3:
            if k_length == 0:
                k_length = k_points_high[2]-k_points_low[2]
            elif k_length != k_points_high[2]-k_points_low[2]:
                if (k1):
                    print("Axis 3 does not match Axis 1 in length")
                    return None
                elif k2:
                    print("Axis 3 does not match Axis 2 in length")
                    return None
        if k_length == 0:
            print("programmer error")
            return None
        data_2D = np.zeros((data_array.shape[0],k_length+1))
        k1_i = k_points_low[0]
        k2_i = k_points_low[1]
        k3_i = k_points_low[2]
        for k in range(0,k_length+1):
            if k1:
                if ascending[0]:
                    k1_i = k + k_points_low[0]
                else:
                    k1_i = k_points_high[0] - k 
            if k2:
                if ascending[1]:
                    k2_i = k + k_points_low[1]
                else:
                    k2_i = k_points_high[1] - k
            if k3:
                if ascending[2]:
                    k3_i = k + k_points_low[2]
                else:
                    k3_i = k_points_high[2] - k
            # print((k1_i,k2_i))
            data_2D[:,k] = data_array[:,k1_i,k2_i,k3_i]
        return data_2D


    # Creates a rough 2D plot for a our Data. Can append Density of States to RHS of plot
    # data_2D      : Data to plot (n_omega,n_k)
    # omegas       : omega values (n_omega)
    # filename     : output file name. Will be in output folder
    # extent       : 4 tuple of k_min,k_max,w_min,w_max
    # x_label      : Label for x axis
    # y_label      : Label for y axis
    # xtick_labels : (2), array of two tuples position of x labels and the x labels
    # Title        : title for the whole figure
    # density_of_states : (n_omega, 2) If passed plots a line plot on RHS showing DOS 
    def plot_heatmap(self,data_2D,omegas,filename,extent=None,
                     x_label="",y_label="$\omega$",xtick_labels="default",title="",
                     density_of_states=None):
        if type(density_of_states) != type(None):
            if extent == None:
                extent = -1, 1, min(omegas), max(omegas)
            fig= plt.figure()
            gs = fig.add_gridspec(1,2,wspace=0,width_ratios=[6,1])
            (ax1,ax2) = gs.subplots(sharey='all')
            fig.suptitle(title)
            ax1.set(xlabel=x_label,ylabel=y_label)
            
            if xtick_labels != "default":
                ax1.set_xticks(xtick_labels[0],xtick_labels[1])
            ax2.plot(density_of_states[:,1],density_of_states[:,0])
            ax2.set_xticks([])
            ax1.imshow(data_2D,origin='lower',extent=extent,cmap='hot',aspect='auto',interpolation='none')
            plt.savefig(output_folder+"/"+filename,bbox_inches='tight')
            ax1.imshow(data_2D,origin='lower',extent=extent,cmap='hot',aspect='auto')
            plt.savefig(output_folder+"/i_"+filename,bbox_inches='tight')
        else:
            plt.xlabel(x_label,fontsize=20)
            plt.ylabel(y_label,fontsize=20)
            plt.title(title,fontsize=20)
            if extent == None:
                extent = -1, 1, min(omegas), max(omegas)

            output_folder = self._out_directory
            if xtick_labels != "default":
                plt.xticks(xtick_labels[0],xtick_labels[1])
            plt.imshow(data_2D,origin='lower',extent=extent,cmap='hot',aspect='auto',interpolation='none')
            plt.savefig(output_folder+"/"+filename,bbox_inches='tight')
            plt.imshow(data_2D,origin='lower',extent=extent,cmap='hot',aspect='auto')
            plt.savefig(output_folder+"/i_"+filename,bbox_inches='tight')
        # plt.plot(density_of_states[:,1],density_of_states[:,0])
        # plt.show()
        return
    

    # Simplifies taking high symmetry cuts in 2D
    #  Gamma -> X - > M -> Gamma
    # data_array : 4D array (n_omega,k1,k2,k3)
    # omegas     : 1D array of omegas
    #
    # Returns data (n_omega,k1), extent, labels
    def high_symmetry_2D(self,data_array,omegas):
        n_k_2 = int((self._n_k_loop[0]+1)/2)
        data_2D = np.zeros((data_array.shape[0],3*n_k_2))
        data_2D[:,0:n_k_2] = self.make_1D(data_array,k_points_low=(0,0,0),k_points_high=(n_k_2-1,0,0),ascending=(True,True,True))
        data_2D[:,n_k_2:2*n_k_2] = self.make_1D(data_array,k_points_low=(n_k_2-1,0,0),k_points_high=(n_k_2-1,n_k_2-1,0),ascending=(True,True,True))
        data_2D[:,2*n_k_2:] = self.make_1D(data_array,k_points_low=(0,0,0),k_points_high=(n_k_2-1,n_k_2-1,0),ascending=(False,False,True))
        extent = 0, 2, min(omegas), max(omegas)
        xticks = (0,.66,1.33,2)
        xtick_labels = ("$\Gamma$","X","M","$\Gamma$")
        return data_2D, extent, [xticks,xtick_labels]
    

    # Gets the density of states
    #   DOS \propto G(r=0,omega)
    # orbital : string in "0+3" format
    def get_DOS(self,orbital):
        
        try:
            file_name = 'greens_up_position_o' + str(orbital) + '_0_0_0.csv'
            density_of_states = np.loadtxt(self._directory+ "/ana_cont/" + file_name,delimiter=' ')
        except:
            print("Couldn't load DOS from ",file_name)
            return None
        
        
        return density_of_states