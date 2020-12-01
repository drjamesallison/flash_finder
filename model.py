import numpy as np
from scipy import special
from scipy import interpolate
from scipy import integrate
from conversions import *

# Define model class
class Model():

    def __init__(self):
        self.input = Input()
        self.output = Output()

    # Calculate the model spectrum
    def calculate_spectrum(self,options,source,cube,nparams):

        # Intialize continuum, emission and absorption components
        self.output.tmp.cont = np.zeros(len(self.input.tmp.x))
        self.output.tmp.emi = np.zeros(len(self.input.tmp.x))
        self.output.tmp.opd = np.zeros(len(self.input.tmp.x))
        self.output.tmp.cal = 1.0
        self.output.tmp.x0_min_emi = 1.e99
        self.output.tmp.x0_max_emi = 0.0
        self.output.tmp.dx_min_emi = 1.e99
        self.output.tmp.dx_max_emi = 0.0
        self.output.tmp.x0_min_opd = 1.e99
        self.output.tmp.x0_max_opd = 0.0
        self.output.tmp.dx_min_opd = 1.e99
        self.output.tmp.dx_max_opd = 0.0

        # Initialize constants object
        from initialize import Constants
        constants = Constants()

        # Loop over user defined model components
        comp_ind = 0
        cube_ind = 0
        y0_before = 1.e99
        for comp in self.input.priors:

            # Continuum component
            if 'continuum' in self.input.types[comp_ind]:

                # Initialize parameters
                cont = 1.0
                c0 = 0.0
                poly = 0.0

                # Loop over parameters
                for prior in self.input.priors[comp_ind]:

                    # Continuum scaling parameter
                    if 'c0' in prior[0]:
                        c0 = cube[cube_ind]

                    # Continuum polynomial parameter
                    if 'fc' in prior[0]:
                        poly += 1.0
                        cont += cube[cube_ind]*np.power((self.input.tmp.x/source.spectrum.x.data[0])-1, poly+1)

                    # Increment cube index
                    cube_ind += 1

                # Multiply continuum component by scaling parameter
                cont *= c0

                # Accumulate continuum component
                self.output.tmp.cont += cont

            # Emission component
            if ('emission' in self.input.types[comp_ind]) and (nparams > self.input.cont_ndims):

                # Initialize parameters
                line  = 0.0
                x0 = 0.0
                dx = 0.0
                y0 = 0.0
                busyb = 0.0
                busyc = 0.0
                busyoff = 0.0

                # Loop over parameters
                for prior in self.input.priors[comp_ind]:

                    # Position parameter
                    if 'x0' in prior[0]:
                        x0 = cube[cube_ind]
                        if x0 < self.output.tmp.x0_min_emi:
                            self.output.tmp.x0_min_emi = x0
                        if x0 > self.output.tmp.x0_max_emi:
                            self.output.tmp.x0_max_emi = x0

                    # Width parameter
                    if 'dx' in prior[0]:
                        dx = cube[cube_ind]
                        if dx < self.output.tmp.dx_min_emi:
                            self.output.tmp.dx_min_emi = dx
                        if dx > self.output.tmp.dx_max_emi:
                            self.output.tmp.dx_max_emi = dx

                    # Peak parameter
                    if 'y0' in prior[0]:
                        y0 = cube[cube_ind]
                        if options.param_out == 'median': 
                            if y0 > y0_before:
                                self.output.tmp.unphys = True
                            y0_before = y0

                    # Busy function steepness parameter 
                    if 'busyb' in prior[0]:
                        busyb = cube[cube_ind]

                    # Busy function parabolic dip scale parameter
                    if 'busyc' in prior[0]:
                        busyc = cube[cube_ind]

                    # Busy function parabolic dip offset parameter
                    if 'busyoff' in prior[0]:
                        busyoff = cube[cube_ind]

                    # Increment cube index
                    cube_ind += 1

                # Define velocities for component
                rest_vel = shift_frame(self.input.tmp.x, x0)
                rest_vel = zTOvel(rest_vel, 'relativistic')
                rest_vel *= constants.LIGHT_SPEED

                # Gaussian model
                if 'gaussian' in self.input.models[comp_ind]:
                    sigma_x = dx/(2*np.sqrt(2*np.log(2)))
                    line = y0*np.exp(-0.5*(rest_vel/sigma_x)**2)

                # Lorentzian model
                if 'lorentzian' in self.input.models[comp_ind]:
                    line = y0*0.25*(dx**2)/(0.25*dx**2 + rest_vel**2)

                # Busy function model
                if 'busyfunc' in self.input.models[comp_ind]:
                    tmp = (special.erf(busyb*(dx+rest_vel))+1)*(special.erf(busyb*(dx-rest_vel))+1)
                    tmp *= busyc*np.power(np.abs(rest_vel-busyoff),int(options.busyn))+1
                    if np.max(tmp) == 0.0:
                        line = 0.0
                    else:
                        line = y0*tmp/np.max(tmp)

                # Accumulate emission component
                self.output.tmp.emi += line

            # Absorption component
            if ('absorption' in self.input.types[comp_ind]) and (nparams > self.input.cont_ndims):

                # Initialize parameters
                line  = 0.0
                x0 = 0.0
                dx = 0.0
                y0 = 0.0
                busyb = 0.0
                busyc = 0.0
                busyoff = 0.0

                # Loop over parameters
                for prior in self.input.priors[comp_ind]:

                    # Position parameter
                    if 'x0' in prior[0]:
                        x0 = cube[cube_ind]
                        if x0 < self.output.tmp.x0_min_opd:
                            self.output.tmp.x0_min_opd = x0
                        if x0 > self.output.tmp.x0_max_opd:
                            self.output.tmp.x0_max_opd = x0

                    # Width parameter
                    if 'dx' in prior[0]:
                        dx = cube[cube_ind]
                        if dx < self.output.tmp.dx_min_opd:
                            self.output.tmp.dx_min_opd = dx
                        if dx > self.output.tmp.dx_max_opd:
                            self.output.tmp.dx_max_opd = dx

                    # Peak parameter
                    if 'y0' in prior[0]:
                        y0 = cube[cube_ind]
                        if options.param_out == 'median':
                            if y0 > np.abs(y0_before):
                                self.output.tmp.unphys = True
                            y0_before = y0

                    # Busy function steepness parameter 
                    if 'busyb' in prior[0]:
                        busyb = cube[cube_ind]

                    # Busy function parabolic dip scale parameter
                    if 'busyc' in prior[0]:
                        busyc = cube[cube_ind]

                    # Busy function parabolic dip offset parameter
                    if 'busyoff' in prior[0]:
                        busyoff = cube[cube_ind]

                    # Increment cube index
                    cube_ind += 1

                # Define velocities for component
                rest_vel = shift_frame(self.input.tmp.x, x0)
                rest_vel = zTOvel(rest_vel, 'relativistic')
                rest_vel *= constants.LIGHT_SPEED

                # Gaussian model
                if 'gaussian' in self.input.models[comp_ind]:
                    sigma_x = dx/(2*np.sqrt(2*np.log(2)))
                    line = y0*np.exp(-0.5*(rest_vel/sigma_x)**2)

                # Lorentzian model
                if 'lorentzian' in self.input.models[comp_ind]:
                    line = y0*0.25*(dx**2)/(0.25*dx**2 + rest_vel**2)

                # Busy function model
                if 'busyfunc' in self.input.models[comp_ind]:
                    tmp = (special.erf(busyb*(dx+rest_vel))+1)*(special.erf(busyb*(dx-rest_vel))+1)
                    tmp *= busyc*np.power(np.abs(rest_vel-busyoff),options.busyn)+1
                    if np.max(tmp) == 0.0:
                        line = 0.0
                    else:
                        line = y0*tmp/np.max(tmp)

                # Accumulate absorption component
                self.output.tmp.opd += line

            # Calibration component
            if 'calibration' in self.input.types[comp_ind]:

                # Loop over parameters
                for prior in self.input.priors[comp_ind]:

                    # Calibration error parameter
                    if 'cal' in prior[0]:
                        cal = cube[cube_ind]
                        self.output.tmp.cal = cal

            # Increment model component index
            comp_ind += 1
        
        # Check for unphysical model
        if np.any(self.output.tmp.cont < 0.) or np.any((1.-np.exp(-1*self.output.tmp.opd)) > 1.):
            self.output.tmp.unphys = True

        return self

    # Calculate the model data
    def calculate_data(self,options,source):

        # Calculate model data
        self.output.tmp.data = self.output.tmp.cont + self.output.tmp.emi

        # Add absorption component to model data
        cont = 1.
        if 'continuum' in self.input.types:
            cont = self.output.tmp.cont
        elif options.y_units != 'abs':
            cont = source.info['flux']
        self.output.tmp.data -= cont*(1.-np.exp(-1*self.output.tmp.opd))

        # Apply channel function
        if options.channel_function != 'none':
            ledge = self.output.tmp.data[0]
            hedge = self.output.tmp.data[-1]
            chanfunc = source.spectrum.y.chanfunc[1,:]
            chanfunc = np.append(chanfunc[-1:0:-1],chanfunc)
            self.output.tmp.data = np.convolve(chanfunc,self.output.tmp.data,mode='same')
            self.output.tmp.data = self.output.tmp.data[0::source.spectrum.x.chansamp]
            self.output.tmp.data /= chanfunc.sum()
            self.output.tmp.data[0] = ledge
            self.output.tmp.data[-1] = hedge
            
        # Apply calibration error (either 1.0 or nuisance parameter)
        self.output.tmp.data *= self.output.tmp.cal

        return self

    # Calculate model astrophysical quantities
    def calculate_astro(self,options,source,cube):

        # Initialize constants object
        from initialize import Constants
        constants = Constants()

        # Initialize index
        ind = self.input.all_ndims

        # Emission properties
        if 'emission' in self.input.types:

            # Set fine grid  
            dfine = self.output.tmp.dx_min_emi/10.0
            dfine /= constants.LIGHT_SPEED
            fine_lim = 2.0*self.output.tmp.dx_max_emi
            fine_lim /= constants.LIGHT_SPEED
            min_fine = self.output.tmp.x0_min_emi - fine_lim
            max_fine = self.output.tmp.x0_max_emi + fine_lim
            x_fine = np.arange(min_fine,max_fine,dfine)

            # Interpolate spectral models to fine grid
            femi = interpolate.interp1d(self.input.tmp.x, self.output.tmp.emi,
                    bounds_error=False, fill_value=0)
            emi_fine = femi(x_fine)

            # Calculate required properties
            peak_S = np.max(emi_fine)
            if peak_S == 0.0:
                peak_z = 0.0
                mean_z = 0.0
                int_S = 0.0
                width = 0.0
            else:
                peak_z = np.mean(x_fine[emi_fine == peak_S])
                mean_z = np.sum(emi_fine*x_fine)/np.sum(emi_fine)
                rest_z = shift_frame(x_fine,mean_z)
                rest_vel = zTOvel(rest_z, 'relativistic')
                rest_vel *= constants.LIGHT_SPEED
                int_S = np.abs(integrate.simps(emi_fine, rest_vel))
                width = int_S/peak_S
            if options.x_units == 'optvel':
                peak_z *= constants.LIGHT_SPEED
                mean_z *= constants.LIGHT_SPEED  

            # Allocate cube
            cube[ind] = np.round(peak_z,99)
            cube[ind+1] = np.round(peak_S,99)
            cube[ind+2] = np.round(int_S,99)
            cube[ind+3] = np.round(width,99)

            # Increment index
            ind += 4

        if 'absorption' in self.input.types:

            # Set fine grid  
            dfine = self.output.tmp.dx_min_opd/10.0
            dfine /= constants.LIGHT_SPEED
            fine_lim = 2.0*self.output.tmp.dx_max_opd
            fine_lim /= constants.LIGHT_SPEED
            min_fine = self.output.tmp.x0_min_opd - fine_lim
            max_fine = self.output.tmp.x0_max_opd + fine_lim
            x_fine = np.arange(min_fine,max_fine,dfine)

            # Interpolate spectral models to fine grid
            fopd = interpolate.interp1d(self.input.tmp.x, self.output.tmp.opd,
                    bounds_error=False, fill_value=0)
            opd_fine = fopd(x_fine)

            # Calculate required properties
            peak_opd = np.max(opd_fine)
            if peak_opd == 0.0:
                peak_z = 0.0
                mean_z = 0.0
                int_opd = 0.0
                width = 0.0
            else:
                peak_z = np.mean(x_fine[opd_fine == peak_opd])
                mean_z = np.sum(opd_fine*x_fine)/np.sum(opd_fine)
                rest_z = shift_frame(x_fine, mean_z)
                rest_vel = zTOvel(rest_z, 'relativistic')
                rest_vel *= constants.LIGHT_SPEED
                int_opd = np.abs(integrate.simps(opd_fine, rest_vel))
                width = int_opd/peak_opd
            if options.x_units == 'optvel':
                peak_z *= constants.LIGHT_SPEED
                mean_z *= constants.LIGHT_SPEED 

            # Allocate cube 
            cube[ind] = np.round(peak_z,99)
            cube[ind+1] = np.round(peak_opd,99)
            cube[ind+2] = np.round(int_opd,99)
            cube[ind+3] = np.round(width,99)

        return

# Define input class
class Input():
    def __init__(self):
        self.tmp = TMP()
        self.types = []
        self.models = []
        self.priors = []
        self.all_ndims = 0
        self.cont_ndims = 0
        self.nparams = 0
        
    # Define method for loading model from file
    def generate_model(self,options,source):

        # Read file
        f = open(options.model_path, 'r')
        while 1:
            lines = f.readlines()
            if not lines:
                break
            for line in lines:            
                
                # Check comments
                if '#' in line:
                    continue
                
                # Assign component type
                comp_typ = line.split()[1].strip()
                self.types.append(comp_typ)

                # Assign parametrization type
                paramz_typ = line.split()[2].strip()
                self.models.append(paramz_typ)            

                # Assign parameter priors
                priors = line.split()[3:]
                
                # Accumulate number of model dimensions
                if 'continuum' in comp_typ or 'calibration' in comp_typ:
                    self.cont_ndims += len(priors)
                self.all_ndims += len(priors)

                # Loop over parameter priors
                ind = 0
                for prior in priors:
                
                    # Strip prior array to desired format
                    prior = prior.strip('[')
                    prior = prior.strip(']')
                    prior = prior.split(',')

                   # Set automated values line position prior
                    if 'x0' in prior[0]:
                            
                        if 'linear' in prior[1] or 'log' in prior[1]:

                            if 'auto' in prior[2]:  
                                prior[2] = min(source.spectrum.x.data)
                            elif 'source' in prior[2]:
                                prior[2] = source.info['z'] - \
                                            abs(float(options.x0_sigma))
                                if options.x_units == 'frequency':
                                    prior[2] = freqTOz(prior[2], options.rest_frequency)
                                elif options.x_units == 'optvel':
                                    prior[2] /= constants.LIGHT_SPEED

                            if 'auto' in prior[3]:
                                prior[3] = max(source.spectrum.x.data)
                            elif 'source' in prior[3]:
                                prior[3] = source.info['z'] + \
                                            abs(float(options.x0_sigma))            
                                if options.x_units == 'frequency':
                                    prior[3] = freqTOz(prior[3], options.rest_frequency)
                                elif options.x_units == 'optvel':
                                    prior[3] /= constants.LIGHT_SPEED

                        if 'normal' in prior[1]:

                            if 'source' in prior[2]:
                                prior[2] = source.info['z']

                            if 'auto' in prior[3]:
                                prior[3] = options.x0_sigma
                            elif 'source' in prior[3]:
                                prior[3] = source.info['e_z']

                    # Set automated values for line peak prior
                    if 'y0' in prior[0]:

                        if 'auto' in prior[2]:

                            prior[2] = 1.e-2*np.min(source.spectrum.y.sigma)

                            if ('absorption' in comp_typ) and (options.y_units != 'abs') and (float(source.info['flux']) != 0):

                                    prior[2] /= float(source.info['flux'])
                                    
                        if 'auto' in prior[3]:

                            if 'absorption' in comp_typ:

                                prior[3] = 1.e2 # A large optical depth value

                                if (options.y_units != 'abs') and (float(source.info['flux']) > float(options.flux_limit)):
                                        
                                        prior[3] = -1.*np.log(1.-float(options.flux_limit)/float(source.info['flux']))

                            elif 'emission' in comp_typ:
                                
                                prior[3] = float(options.flux_limit)

                    # Set automated values for Busy function steepness parameter
                    if 'busyb' in prior[0]:
                        
                        if 'auto' in prior[2]:
                            prior[2] = 1.e-99

                        if 'auto' in prior[3]:
                            prior[3] = 1.e99

                    # Set automated values for Busy function parabolic dip scale parameter
                    if 'busyc' in prior[0]:

                        if 'auto' in prior[2]:
                            prior[2] = 1.e-99

                        if 'auto' in prior[3]:
                            prior[3] = 1.e99

                    # Busy function parabolic dip offset parameter
                    if 'busyoff' in prior[0]:
                        
                        if 'auto' in prior[2]:
                            prior[2] = -1.e2

                        if 'auto' in prior[3]:
                            prior[3] = 1.e2

                    # Set automated values for continuum scaling parameter
                    if 'c0' in prior[0]:

                        if 'auto' in prior[2]:
                            prior[2] = 1.e-1*float(source.info['flux'])

                        if 'auto' in prior[3]:
                            prior[3] = 10.0*float(source.info['flux'])

                    # Set automated values for continuum component parameter
                    if 'fc' in prior[0]:

                        if 'auto' in prior[2]:
                            prior[2] = -1.0

                        if 'auto' in prior[3]:
                            prior[3] = 1.0

                    # Update priors
                    priors[ind] = prior        

                    # Increment index
                    ind += 1

                # Accumulate model priors
                self.priors.append(priors)

        # Update number of model parameters
        self.nparams = self.all_ndims
        if 'emission' in self.types:
            self.nparams += 4
        if 'absorption' in self.types:
            self.nparams += 4

        return self

# Define output class
class Output():
    def __init__(self):
        self.nmodes = 0
        self.ndetections = 0
        self.tmp = TMP()
        self.null = Null()
        self.sline = Sline()
        self.cont = Cont()

# Define null statistic class 
class Null():
    def __init__(self):
        self.logZ = -1.e999

# Define measured qauntity class
class Quantity():
    def __init__(self):
        mean = 0
        sigma = 0

# Define spectral line model class
class Sline():
    def __init__(self):
        self.nmodes = 0
        self.logZ = Quantity()
        self.params = Params()
        self.weights = Weights()
        self.loglhood = Loglhood()


# Define continuum model class
class Cont():
    def __init__(self):
        self.logZ = Quantity()
        self.params = Params()
        self.weights = Weights()
        self.loglhood = Loglhood()

# Define model parameters class
class Params():
    def __init__(self):
        self.mean = []
        self.sigma = []
        self.maxl = []
        self.maxp = []
        self.median = []
        self.onesigma = []
        self.twosigma = []
        self.threesigma = []
        self.fivesigma = []
        self.samples = []

# Define weights class
class Weights():
    def __init__(self):
        self.cum = []
        self.samples = []

# Define likelihood class
class Loglhood():
    def __init__(self):
        self.samples = []

# Define best fitting spectrum class
class Maxlspect():
    def __init__(self):
        self.data = Data()
        self.chisq = Quantity()

# Define data class
class Data():
    def __init__(self):
        self.cont = np.array([])
        self.line = np.array([])
        self.total = np.array([])

# Define temporary dummy class
class TMP():
    def __init__(self):
        pass

