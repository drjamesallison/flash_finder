# Set path to FLASH FINDER
FINDER=$HOME/software/flash_finder/ 
export FINDER

# Set path to MULTINEST
export MULTINEST=$HOME/software/MultiNest_v3.10/

# Set path to PYMULTINEST
export PYMULTINEST=$HOME/software/pymultinest/

# Add MultiNest library to dynamic library path
export DYLD_LIBRARY_PATH=$MULTINEST/lib:${DYLD_LIBRARY_PATH}
# export LD_LIBRARY_PATH=$MULTINEST/lib:${LD_LIBRARY_PATH}

# Set path to Matplotlib set up
export MATPLOTLIBRC=$FINDER/matplotlib/
