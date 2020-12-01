import numpy as np
from scipy import special 
from model import *
from initialize import *

# Calculate fitting statistic
def calculate_chisquared(options,data,model):    
    if options.corr_switch:
        chisq = (data.y.data.view(np.matrix)-model.view(np.matrix))* \
                data.y.invcov*np.transpose(data.y.data.view(np.matrix)- \
                        model.view(np.matrix))
    else:
        chisq = np.sum(((data.y.data-model)/data.y.sigma)**2)

    return chisq

# Prior function supplied to MultiNest
def PriorGen(options,model):
    def Prior(cube,ndim,nparams):
        
        # Loop over components and parameters
        cube_ind = 0
        comp_ind = 0

        for comp in model.input.priors:
            if (nparams == model.input.cont_ndims) and ('continuum' not in model.input.types[comp_ind]) \
                and ('calibration' not in model.input.types[comp_ind]):
                pass

            else:
                for prior in comp:

                    ptype = str(prior[1])
                    pmin = float(prior[2])
                    pmax = float(prior[3])

                    # Convert unit sampled value to desired physical coordinates
                    if ptype == 'linear':
                        cube[cube_ind] *= pmax - pmin
                        cube[cube_ind] += pmin
                    if ptype == 'log':
                        lmin = np.log(np.abs(pmin))
                        lmax = np.log(np.abs(pmax))
                        cube[cube_ind] *= lmax - lmin
                        cube[cube_ind] += lmin
                        cube[cube_ind] = np.exp(cube[cube_ind])
                        if pmin < 0.0 or pmax < 0.0:
                            cube[cube_ind] *= -1.0
                    elif ptype == 'normal':
                        cube[cube_ind] *= 2.0
                        cube[cube_ind] -= 1.0
                        cube[cube_ind] = special.erfinv(cube[cube_ind])
                        cube[cube_ind] *= pmax*np.sqrt(2)
                        cube[cube_ind] += pmin

                    # Increment cube index
                    cube_ind += 1

            # Increment component index
            comp_ind += 1

        return
    return Prior

# Log-likelihood function supplied to MultiNest
def LogLGen(options,source,model):
    def LogL(cube,ndim,nparams):

        # Initialize model data x-axis
        model.input.tmp.x = np.copy(source.spectrum.x.fine)
    
        # Calculate model spectrum
        model.output.tmp.unphys = False
        model.calculate_spectrum(options,source,cube,nparams)

        # Return with low likelihood if unphysical model
        if model.output.tmp.unphys == True:
            lnew = -1.e999
            return lnew 

        # Calculate model data
        model.calculate_data(options,source)

        # Calculate astronomical quantities
        if (nparams > model.input.cont_ndims):
            model.calculate_astro(options,source,cube)

        # Calculate chi-squared
        chisq = calculate_chisquared(options,source.spectrum,model.output.tmp.data)   
        
        # Calculate log-likelihood
        lnew = -0.5*chisq
        lnew -= model.output.null.logZ
        
        return lnew
    return LogL




