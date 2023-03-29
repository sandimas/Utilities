#!/bin/bash
#This file is a submission script to request the ISAAC resources from Slurm 
#SBATCH -J AC			       #The name of the job
#SBATCH -A ACF-UTK0014              # The project account to be charged
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1          # cpus per node 
#SBATCH --partition=condo-sjohn145            # If not specified then default is "campus"
#SBATCH --time=0-24:00:00             # Wall time (days-hh:mm:ss)
#SBATCH --error=run_logs/AC.e%J	       # The file where run time errors will be dumped
#SBATCH --output=run_logs/AC.o%J	       # The file where the output of the terminal will be dumped
#SBATCH --qos=condo

 # Now list your executable command/commands.
 # Example for code compiled with a software module:
echo "ORBITALS " ${ORBITALS} $ORBITALS
python3 ana_cont.py ${FOLDER} ${CORRELATION} ${SPACE} ${OMEGA_MIN} ${OMEGA_MAX} ${N_OMEGA} ${K_MIN} ${K_MAX} ${ORBITALS} ${ALPHA} ${OPTIMIZER} ${NAMETAG}

