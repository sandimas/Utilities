FOLDER="hubbard_cuprate_beta32.00_u-2.050-1"
CORRELATION="density"
OMEGA_MIN=0.1   # For greens function -5:15 is generally good
OMEGA_MAX=10.0  # for spin_z and density 0:10 is generally good
                # For spin structure factor 0:2 will often fail  
N_OMEGA=200
ORBITALS="0+1+2+3"
ALPHA="chi2kink" # For density and greens chi2kink is preferred. spin_z might require classic
OPTIMIZER="scipy_lm"
NAMETAG=""  # added to csv files so you can run different parameter sets
            # without overwriting data
SPACE="momentum" # "momentum" or "position"


# From the python script:
# python3 ana_cont.py FOLDER CORRELATION OMEGA_MIN OMEGA_MAX N_OMEGA K_MIN K_MAX ALPHA OPTIMIZER NAMETAG
#
# examples:
#  1D Chain, k points [0,19]
#    python3 ana_cony.py "/path_to/hubbard_cuprate_n1.00_n-1.450-1" "spin_z" \
#                         -10.0 10.0 101 0 19 "chi2kink" "scipy_lm" "chi2run"
#
#  1D Chain, default settings, all k points
#    python3 ana_cony.py "/path_to/hubbard_cuprate_n1.00_n-1.450-1" "greens_up" \
#                         -10.0 10.0 101
#
#  3D, passing 3D k points with k_x = 1, k_y = [0,19], k_z = 4
#    python3 ana_cony.py "/path_to/hubbard_cuprate_n1.00_n-1.450-1" "spin_z" \
#                         -10.0 10.0 101 1,0,4 1,19,4
#
# SEE ana_cont.py for more documentation

         

for K1  in $(seq 0 1 19)
do
    sbatch --export=FOLDER=${FOLDER},CORRELATION=${CORRELATION},OMEGA_MIN=${OMEGA_MIN},OMEGA_MAX=${OMEGA_MAX},N_OMEGA=${N_OMEGA},K_MIN=${K1},K_MAX=$((K1)),ORBITALS=${ORBITALS},ALPHA=${ALPHA},OPTIMIZER=${OPTIMIZER},NAMETAG=${NAMETAG} AC.slurm
done

