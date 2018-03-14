# Set path to FLASH FINDER
ACES=$HOME/software/ACES/
FINDER=$ACES/habs_nest/ 
export FINDER

# Set path to MULTINEST
export MULTINEST=$HOME/software/MultiNest_v3.10/

# Add MultiNest library to dynamic library path
export DYLD_LIBRARY_PATH=$MULTINEST/lib:${DYLD_LIBRARY_PATH}
# export LD_LIBRARY_PATH=$MULTINEST/lib:${LD_LIBRARY_PATH}

# Set path to Matplotlib set up
export MATPLOTLIBRC=$HABSNEST/matplotlib/
