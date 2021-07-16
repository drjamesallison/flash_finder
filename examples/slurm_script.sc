#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --job-name=flashfinder_test
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --account=ja3

module load python
module load argparse
module load numpy
module load matplotlib
module load astropy
module unload PrgEnv-cray/6.0.4
module load mpi4py
module unload gcc
module load scipy
module use /group/askap/jallison/multinest/modulefiles
module load multinest
module use /group/askap/jallison/pymultinest/modulefiles
module load pymultinest
module use /group/askap/jallison/corner/modulefiles
module load corner
module use /group/askap/jallison/flash_finder/modulefiles
module load flash_finder

export MPICH_GNI_MALLOC_FALLBACK=enabled

srun --export=ALL --ntasks=20 --ntasks-per-node=20 python $FINDER/flash_finder.py \
--x_units 'optvel' \
--y_units 'mJy' \
--plot_switch \
--out_path $FINDER'/examples/chains' \
--data_path $FINDER'/examples/data/' \
--nlive 500 \
--channel_function 'none' \
--plot_restframe 'none' \
--noise_factor '1.00' \
--mmodal \
--mpi_switch \
